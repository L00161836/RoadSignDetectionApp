import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend as K

# Setting up the directory path for the gathered images
imageDirectory = pathlib.Path('input')

# Displays the amount of images in the dataset
imageCount = len(list(imageDirectory.glob('**/*.png')))
print(imageCount)

# Define the standardised image size and batch size (amount of images the model will train with at a time).
batchSize = 32
imageSize = (180, 180)

# Creating the training dataset from 80% of the overall dataset using the parameters set above.
trainingDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=imageSize,
    batch_size=batchSize
)

# Creating the testing dataset from 20% of the overall dataset using the parameters set above.
validationDataset = tf.keras.utils.image_dataset_from_directory(
    imageDirectory,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=imageSize,
    batch_size=batchSize
)

classNames = trainingDataset.class_names

validationBatches = tf.data.experimental.cardinality(validationDataset)
testDataset = validationDataset.take(validationBatches // 5)
validationDataset = validationDataset.skip(validationBatches // 5)

autotune = tf.data.AUTOTUNE

trainingDataset = trainingDataset.cache().prefetch(buffer_size=autotune)
validationDataset = validationDataset.cache().prefetch(buffer_size=autotune)
testDataset = testDataset.cache().prefetch(buffer_size=autotune)

dataAugmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomBrightness(0.5),
    tf.keras.layers.RandomContrast(0.5)
])

# for image, _ in trainingDataset.take(1):
#     plt.figure(figsize=(10, 10))
#     first_image = image[0]
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         augmented_image = dataAugmentation(tf.expand_dims(first_image, 0))
#         plt.imshow(augmented_image[0] / 255)
#         plt.axis('off')
#
#     plt.show()

preprocessInput = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

imageShape = imageSize + (3,)
baseModel = tf.keras.applications.MobileNetV2(input_shape=imageShape,
                                              include_top=False,
                                              weights='imagenet')

imageBatch, labelBatch = next(iter(trainingDataset))
featureBatch = baseModel(imageBatch)
print(featureBatch.shape)

baseModel.trainable = False

baseModel.summary()

globalAverageLayer = tf.keras.layers.GlobalAveragePooling2D()
featureBatchAverage = globalAverageLayer(featureBatch)
print(featureBatchAverage.shape)

predictionLayer = tf.keras.layers.Dense(3, activation='relu')
predictionBatch = predictionLayer(featureBatchAverage)
print(predictionBatch.shape)

inputs = tf.keras.Input(shape=(180, 180, 3))
x = dataAugmentation(inputs)
x = preprocessInput(x)
x = baseModel(x, training=False)
x = globalAverageLayer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = predictionLayer(x)
model = tf.keras.Model(inputs, outputs)

baseLearningRate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=baseLearningRate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

print(len(model.trainable_variables))

initialEpochs = 25

loss0, accuracy0 = model.evaluate(validationDataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(trainingDataset,
                    epochs=initialEpochs,
                    validation_data=validationDataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.5])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

baseModel.trainable = True

fineTuneAt = 100

for layer in baseModel.layers[:fineTuneAt]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=baseLearningRate/10),
              metrics=['accuracy'])

fineTuneEpochs = 25
totalEpochs = initialEpochs + fineTuneEpochs

historyFine = model.fit(trainingDataset,
                        epochs=totalEpochs,
                        initial_epoch=history.epoch[-1],
                        validation_data=validationDataset)

acc += historyFine.history['accuracy']
val_acc += historyFine.history['val_accuracy']

loss += historyFine.history['loss']
val_loss += historyFine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.3, 1])
plt.plot([initialEpochs-1,initialEpochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.3])
plt.plot([initialEpochs-1,initialEpochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, accuracy = model.evaluate(testDataset)
print('Test accuracy :', accuracy)

imageBatch, labelBatch = testDataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(imageBatch).flatten()

predictions = tf.nn.softmax(-1, 0., 1.)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', labelBatch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(imageBatch[i].astype("uint8"))
  plt.title(classNames[predictions[i]])
  plt.axis("off")


# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(3)
# ])
#
# model.compile(
#   optimizer='adam',
#   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])
#
# history = model.fit(
#   trainingDataset,
#   validation_data=validationDataset,
#   epochs=15
# )
# #
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower right')
#
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim([0, 0.6])
# plt.legend(['Training Loss', 'Testing Loss'], loc='lower right')
#
# plt.show()

