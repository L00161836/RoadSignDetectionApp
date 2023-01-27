import numpy as np
import os
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

imageDirectory = pathlib.Path('input')
imageCount = len(list(imageDirectory.glob('**/*.png')))
print(imageCount)

fiftyKphImages = list(imageDirectory.glob('50kph/*'))
print(fiftyKphImages[0])
image = PIL.Image.open(str(fiftyKphImages[0]))

batchSize = 32
imageHeight = 180
imageWidth = 180

trainingDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

validationDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

classNames = trainingDataset.class_names
print(classNames)

plt.figure(figsize=(10, 10))
for images, labels in trainingDataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classNames[labels[i]])
        plt.axis("off")

plt.show()


