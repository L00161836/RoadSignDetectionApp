import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

# Setting up the directory path for the gathered images
imageDirectory = pathlib.Path('input')

# Displays the amount of images in the dataset
imageCount = len(list(imageDirectory.glob('**/*.png')))
print(imageCount)

# Define the standardised image size and batch size (amount of images the model will
# train with at a time).
batchSize = 32
imageHeight = 180
imageWidth = 180

# Creating the training dataset from 80% of the overall dataset using the parameters set above
trainingDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(imageHeight, imageWidth),
    batch_size=batchSize
)

# Creating the testing dataset from 20% of the overall dataset using the parameters set above
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


