import pathlib
import tensorflow as tf

# Setting up the directory path for the gathered images
imageDirectory = pathlib.Path('input')

# Displays the amount of images in the dataset
imageCount = len(list(imageDirectory.glob('**/*.png')))
print(imageCount)

# Define the standardised image size and batch size (amount of images the model will train with at a time).
batchSize = 16
imageSize = (240, 240)

# Creating the training dataset from 80% of the overall dataset using the parameters set above.
trainingDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=imageSize,
    batch_size=batchSize,
    color_mode="grayscale"
)

# Creating the testing dataset from 20% of the overall dataset using the parameters set above.
validationDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=imageSize,
    batch_size=batchSize,
    color_mode="grayscale"
)

# Autotune will dynamically alter the buffer size of the model depending on current resources available.
autotune = tf.data.AUTOTUNE
trainingDataset = trainingDataset.cache().prefetch(buffer_size=autotune)
validationDataset = validationDataset.cache().prefetch(buffer_size=autotune)

# Building the model using the data augmentation layers and the imported mature model as an added layer
model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(240, 240, 1)),
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.applications.MobileNetV2(input_shape=(240, 240, 1),
                                      include_top=False,
                                      weights='imagenet'),
    tf.keras.layers.Conv2D(128, 3, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64, 3, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(32, 3, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
])
# Condenses everything down to an array of three floats, the prediction results.
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

# Outputs a summary of all layers within the model
model.summary()

# Compiling the model.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model.
model.fit(
    trainingDataset,
    validation_data=validationDataset,
    epochs=100
)

# Converting the model to TF Lite for mobile application usage.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfliteModel = converter.convert()

# Saving the TF Lite model.
with open("model.tflite", "wb") as f:
    f.write(tfliteModel)
