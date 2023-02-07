import pathlib
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# Setting up the directory path for the gathered images
imageDirectory = pathlib.Path('input')

# Displays the amount of images in the dataset
imageCount = len(list(imageDirectory.glob('**/*.png')))
print(imageCount)

# Define the standardised image size and batch size (amount of images the model will train with at a time).
batchSize = 32
imageHeight = 32
imageWidth = 32

# Creating the training dataset from 80% of the overall dataset using the parameters set above.
trainingDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

# Creating the testing dataset from 20% of the overall dataset using the parameters set above.
validationDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(trainingDataset, epochs=30,
                    validation_data=validationDataset)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

plt.show()

