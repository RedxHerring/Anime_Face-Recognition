import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
import os
import random
import shutil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def classify_image_type(base_dir='datasetsTON',imres=(96,96)):
    # Check if Intel Arc GPU is available
    intel_arc_available = tf.config.list_physical_devices("GPU")
    if intel_arc_available and "ARC" in intel_arc_available[0].name:
        os.environ["TF_ADJUST_HUB_FALLBACK_TIMEOUT"] = "600"  # Increase the timeout for loading TensorFlow Hub models on Intel Arc GPU
        os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "COMPRESSED"  # Set the model load format for TensorFlow Hub

        # Use Intel extension for TensorFlow (oneAPI)
        import tensorflow.oneapi as tfoneapi
        tfoneapi.initialize()
    # Enable GPU acceleration
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    model.save('models/anime_classifier_model.h5')

def get_image_type(image_path,model=None):
    # Load the trained model
    if model is None:
        model = keras.models.load_model('models/anime_classifier_model.h5')

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

def build_dataset_recursive(base_dir, imres, subset):
    return tf.keras.preprocessing.image_dataset_from_directory(
        base_dir,
        validation_split=0,
        label_mode="categorical",
        seed=321,
        image_size=imres,
        batch_size=1)

class_names = None

def train_face_recognition(base_dir='datasets_recursive',imres=(96,96)):
    batch_size = 16
    checkpoint_path = "checkpt"
    training_set = build_dataset_recursive(base_dir, imres, "training")
    class_names = tuple(training_set.class_names)
    training_size = training_set.cardinality().numpy()
    training_set = training_set.unbatch().batch(batch_size)
    training_set = training_set.repeat()
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    preprocessing_model = tf.keras.Sequential([normalization_layer])
    do_data_augmentation = True
    if do_data_augmentation:
        preprocessing_model.add(
            tf.keras.layers.RandomRotation(40))
        preprocessing_model.add(
            tf.keras.layers.RandomRotation(0.4))
        preprocessing_model.add(
            tf.keras.layers.RandomTranslation(0, 0.2))
        preprocessing_model.add(
            tf.keras.layers.RandomTranslation(0.2, 0))
        preprocessing_model.add(
            tf.keras.layers.RandomZoom(0.2, 0.2))
        preprocessing_model.add(
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"))
        preprocessing_model.add(
            tf.keras.layers.RandomContrast(0.2))
    training_set = training_set.map(lambda images, labels:
                            (preprocessing_model(images), labels))
    # validation_set = build_dataset_recursive(base_dir, imres, "validation")
    # validation_set_all = validation_set
    # validation_size = validation_set.cardinality().numpy()
    # validation_set = validation_set.unbatch().batch(batch_size)
    # validation_set = validation_set.map(lambda images, labels:
    #                     (normalization_layer(images), labels))
    do_fine_tuning = True
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=imres + (3,)),
        hub.KerasLayer('https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1', trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.7),
        tf.keras.layers.Dense(len(class_names),
                            kernel_regularizer=tf.keras.regularizers.l2(0.0003))
    ])
    model.build((None,)+imres+(3,))
    model.summary()
    epoch = 100
    lr = 0.001
    decay_rate = lr / epoch
    model.compile(
        optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=lr, momentum=0.9, nesterov=False, decay=decay_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy',tf.keras.metrics.TopKCategoricalAccuracy(name='top-5-accuracy')],)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                            patience=3, min_lr=0.00001)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    steps_per_epoch = training_size // batch_size
    # validation_steps = validation_size // batch_size
    hist = model.fit(
        training_set,
        epochs=epoch, steps_per_epoch=steps_per_epoch,
        # validation_data=validation_set,
        # validation_steps=validation_steps,
        callbacks=[reduce_lr, checkpoint_callback, callback]).history
    return model, hist

def test_model(model, base_dir='datasets_anime', imres=(96,96)):
    test_set = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        image_size=imres,
        label_mode="categorical",
        batch_size=1)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    test_set = test_set.map(lambda images, labels:
                        (normalization_layer(images), labels))
    _, accuracy, top_5_accuracy = model.evaluate(test_set)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    characters = os.listdir(base_dir)
    for character in characters:
        files = os.listdir(base_dir + '/' + character)
        for image_file in files:
            img_f = base_dir + '/' + character + '/' + image_file
            img = tf.keras.utils.load_img(
                img_f , target_size=imres
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            img_array = normalization_layer(img_array)

            image = img_array[0,:,:,:]

            print(f"True Label: {character}")

            print("This image is likely belongs to:")
            predictions = model.predict(img_array)

            score = tf.nn.softmax(predictions[0])

            predicted_index = np.argsort(score)[-5:][::-1]

            for idx in predicted_index:
                prediction_score = (100 * score[idx])

                print(
                    f"> {class_names[idx]} : {prediction_score:.2f} % confidence."
                )

if __name__ == "__main__":
    # classify_image_type()
    model, hist = train_face_recognition()
    test_model(model)
