import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
import os
import random
import shutil
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import shutil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from utils import files_in_dir

import cv2
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.models import efficientnet_v2_l


def train_image_type(base_dir='datasetsTON',imres=(96,96),out_name='models/image_type_classifier_model.h5'):
    # Set up the CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imres[0], imres[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # Data augmentation to improve generalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    # Define directories
    train_data_dir = os.path.join(base_dir, 'training')
    validation_data_dir = os.path.join(base_dir, 'validation')
    # reset contents of directories
    shutil.rmtree(train_data_dir,ignore_errors=True)
    shutil.rmtree(validation_data_dir,ignore_errors=True)
    # remaining directories are the discrete classes
    class_names = glob.glob(os.path.join(base_dir,'*'))
    # Recreate directory
    os.makedirs(train_data_dir,exist_ok=True)
    for class_dir in class_names:
        shutil.copytree(class_dir,os.path.join(train_data_dir,os.path.basename(class_dir)))

    # Move a fraction of files from train directory to validation directory
    validation_split = 0.05  # Fraction of files to move to validation
    for class_name in os.listdir(train_data_dir):
        class_dir = os.path.join(train_data_dir, class_name)
        files = os.listdir(class_dir)
        num_validation_files = int(validation_split * len(files))
        validation_files = random.sample(files, num_validation_files)
        # Move validation files to validation directory
        for file_name in validation_files:
            src_path = os.path.join(class_dir, file_name)
            dst_path = os.path.join(validation_data_dir, class_name, file_name)
            os.makedirs(os.path.join(validation_data_dir, class_name), exist_ok=True)
            shutil.move(src_path, dst_path)

    # Load and preprocess the dataset
    batch_size = 32
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=imres,
        batch_size=batch_size,
        class_mode='categorical'
    )
    # Validation data generator
    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=imres,
        batch_size=batch_size,
        class_mode='categorical',
    )
    # Train the model
    epochs = 10
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        verbose=1
    )
    # Extract accuracy values from the training history
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    # Print accuracy for each epoch
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} - Training Accuracy: {training_accuracy[epoch]:.4f} - Validation Accuracy: {validation_accuracy[epoch]:.4f}")
    # Save the trained model
    model.save(out_name)
    return model


def classify_image_type(image_path,model=None,model_name='models/image_type_classifier_model.h5'):
    # Load the trained model
    if model is None:
        model = keras.models.load_model(model_name)

    # Define the class labels
    class_labels = sorted(['this_anime', 'other_anime', 'not_anime'])

    # reprocess the new image
    image = load_img(image_path, target_size=(96, 96))
    image = img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, 0)  # Add a batch dimension

    # Make predictions
    predictions = model.predict(image)
    predicted_class_index = tf.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label, model


def create_missing_classes(training_dir,validation_dir):
    training_classes = set(os.listdir(training_dir))
    validation_classes = set(os.listdir(validation_dir))
    missing_classes = training_classes - validation_classes
    for missing_class in missing_classes:
        out_dir = os.path.join(validation_dir, missing_class)
        os.makedirs(out_dir)
        images_list = files_in_dir(os.path.join(training_dir,missing_class))
        img_name = images_list[0]
        img = cv2.flip(cv2.imread(img_name), 1) # load and flip horizontally
        base_name, ext = os.path.splitext(os.path.basename(img_name))
        out_name = os.path.join(out_dir,base_name+'_hflip'+ext)
        cv2.imwrite(out_name,img)


def build_dataset(base_dir='datasets_recursive', imres=(96, 96), subset='training'):
    return tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        validation_split=0,
        label_mode="categorical",
        seed=321,
        image_size=imres,
        batch_size=1)


def train_face_recognition_tf(training_dir='datasets_training', validation_dir='datasets_anime', imres=(96, 96), num_augmented_images=100, out_name='models/saved_model.h5', reg=.01, model=None):
    batch_size = 16
    checkpoint_path = "checkpt"

    create_missing_classes(training_dir=training_dir, validation_dir=validation_dir)

    # Prepare training and validation datasets
    training_set = build_dataset(training_dir, imres, "training")
    class_names = tuple(training_set.class_names)
    training_size = training_set.cardinality().numpy()
    training_set = training_set.unbatch().repeat().batch(batch_size)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    validation_set = build_dataset(validation_dir, subset='validation')
    validation_size = validation_set.cardinality().numpy()
    validation_set = validation_set.unbatch().batch(batch_size)
    validation_set = validation_set.map(lambda images, labels: (normalization_layer(images), labels))

    # Data augmentation
    preprocessing_model = tf.keras.Sequential([normalization_layer])
    preprocessing_model.add(tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(tf.keras.layers.RandomTranslation(0, 0.2))
    preprocessing_model.add(tf.keras.layers.RandomTranslation(0.2, 0))
    preprocessing_model.add(tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(tf.keras.layers.RandomContrast(0.2))
    training_set = training_set.map(lambda images, labels: (
        preprocessing_model(images), labels)).repeat(num_augmented_images)

    do_fine_tuning = True
    if model is None:
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=imres + (3,)),
            hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1', trainable=do_fine_tuning),
            tf.keras.layers.Dropout(rate=0.7),
            tf.keras.layers.Dense(len(class_names),
                                kernel_regularizer=tf.keras.regularizers.l2(reg))
        ])
    else:
        # Update the l2 regularization of the input model
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_regularizer = tf.keras.regularizers.l2(reg)
    num_epochs = 10
    lr = 0.001
    decay_rate = lr / num_epochs
    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(
            learning_rate=lr, momentum=0.9, nesterov=False, decay=decay_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(name='top-5-accuracy')],
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    steps_per_epoch = (training_size * num_augmented_images) // batch_size
    validation_steps = validation_size // batch_size

    hist = model.fit(
        training_set,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_set,
        validation_steps=validation_steps,
        callbacks=[reduce_lr, checkpoint_callback, callback]
    ).history
    model.save(out_name)
    return model, hist


def train_face_recognition_torch(recursive_dir='datasets_recursive', source_dir='datasets_anime', imres=(96, 96), out_name='models/saved_model.pt'):
    batch_size = 16
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import intel_extension_for_pytorch as ipex
            device = 'xpu'
        except: 
            device = "cpu"
    
    create_missing_classes(training_dir=recursive_dir, validation_dir=source_dir)

    # Data transformations
    data_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(imres, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load datasets
    train_dataset = datasets.ImageFolder(root=recursive_dir, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_dataset = datasets.ImageFolder(root=source_dir, transform=data_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define the model
    model = efficientnet_v2_l()
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # intitial train
    model.train()
    ### set to device and continue
    model.to(device)
    criterion.to(device)
    if device == 'xpu':
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss / len(validation_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%")
    # Save the trained model
    torch.save(model.state_dict(), out_name)
    return model

def load_existing_model(model_name='models/saved_model.h5'):
    model = tf.keras.models.load_model(model_name,
       custom_objects={'KerasLayer':hub.KerasLayer})
    return model


def classify_character_tf(character, class_names, imres=(96, 96), source_dir='datasets_anime', out_dir='datasets_recursive', model=None):
    if model is None:
        model = load_existing_model()
    img_files = files_in_dir(os.path.join(source_dir,character)) # get full filenames
    Nf = len(img_files)
    accuracy = np.zeros(Nf)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    print(f"Character: {character}")
    for fidx,img_f in enumerate(img_files):
        img = tf.keras.utils.load_img(
            img_f , target_size=imres
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        img_array = normalization_layer(img_array)
        image = img_array[0,:,:,:]
        predictions = model.predict(img_array)

        score = tf.nn.softmax(predictions[0])

        if score is not None:
            predicted_index = np.argsort(score)[-5:][::-1]
            for cidx in predicted_index:
                if class_names[cidx] == character:
                    accuracy[fidx] = 100*score[cidx]
    print(accuracy)
    histogram, edges = np.histogram(accuracy)
    # finding a decent threshold by looking for local minima and choosing the rightmost one
    local_minima = find_peaks(np.negative(histogram))
    if local_minima[0].size != 0:
        cutoff = edges[local_minima[0][-1]]
        matches = np.flatnonzero(accuracy >= cutoff)
        # plt.stairs(histogram, edges, fill=True)
        # plt.show()
        for match in matches:
            file2 = os.path.join(out_dir,character,os.path.basename(img_files[match]))
            print(f'Copying {img_files[match]} to {file2}')
            shutil.copy(img_files[match],file2)
        return (len(matches)/Nf)
    return 0


def classify_all_characters_tf(in_dir='datasets_anime',out_dir='datasets_recursive',model_name='models/saved_model.h5'):
    training_set = build_dataset(base_dir=in_dir)
    class_names = tuple(training_set.class_names)
    characters = os.listdir(in_dir)
    C = len(characters)
    overall_matches = np.zeros(C)
    model = load_existing_model(model_name)
    for idx,character in enumerate(characters):
        print(f'{idx+1}/{C}')
        detection_rate = classify_character_tf(character, class_names,source_dir=in_dir, out_dir=out_dir, model=model)
        overall_matches[idx] = detection_rate
        print(detection_rate)
        print(overall_matches)
        print(f'Overall detection rate: {np.mean(overall_matches)}')


if __name__ == "__main__":
    # classify_image_type()
    # model, hist = train_face_recognition_tf()
    # model = load_existing_model()
    classify_all_characters()
