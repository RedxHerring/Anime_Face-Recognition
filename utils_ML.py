import tensorflow as tf
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

if __name__ == "__main__":
    classify_image_type()
