import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# import tensorflow_datasets as tfds
from keras import backend as K

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

classNames = trainingDataset.class_names

validationBatches = tf.data.experimental.cardinality(validationDataset)
testDataset = validationDataset.take(validationBatches // 5)
validationDataset = validationDataset.skip(validationBatches // 5)

autotune = tf.data.AUTOTUNE

trainingDataset = trainingDataset.cache().prefetch(buffer_size=autotune)
validationDataset = validationDataset.cache().prefetch(buffer_size=autotune)
testDataset = testDataset.cache().prefetch(buffer_size=autotune)

# dataAugmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip('horizontal_and_vertical'),
#     tf.keras.layers.RandomRotation(0.3),
#     tf.keras.layers.RandomBrightness(0.5),
#     tf.keras.layers.RandomContrast(0.5)
# ])
# #
# preprocessInput = tf.keras.applications.mobilenet_v2.preprocess_input
# rescale = tf.keras.layers.Rescaling(1./255, offset=-1)
#
# imageShape = (180, 180, 3)
# baseModel = tf.keras.applications.MobileNetV2(input_shape=(180, 180, 3),
#                                               include_top=False,
#                                               weights='imagenet')
#
# baseModel.trainable = True
#
# globalAverageLayer = tf.keras.layers.GlobalAveragePooling2D()
#
# predictionLayer = tf.keras.layers.Dense(3, activation='relu')
#
# inputs = tf.keras.Input(shape=(180, 180, 3))
# x = dataAugmentation(inputs)
# x = preprocessInput(x)
# x = baseModel(x, training=False)
# x = globalAverageLayer(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# outputs = predictionLayer(x)
# model = tf.keras.Model(inputs, outputs)
#
# baseLearningRate = 0.0001
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=baseLearningRate),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model.summary()

model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(240, 240, 1)),
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.3),
    # tf.keras.applications.MobileNetV2(input_shape=(240, 240, 1),
    #                                   include_top=False,
    #                                   weights='imagenet'),
    tf.keras.layers.Conv2D(128, 3, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(64, 3, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Conv2D(32, 3, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
])

model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    trainingDataset,
    validation_data=validationDataset,
    epochs=25
)

#
# print(len(model.trainable_variables))
#
# initialEpochs = 100
#
# loss0, accuracy0 = model.evaluate(validationDataset)
#
# print("initial loss: {:.2f}".format(loss0))
# print("initial accuracy: {:.2f}".format(accuracy0))
#
# history = model.fit(trainingDataset,
#                     epochs=initialEpochs,
#                     validation_data=validationDataset)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# baseModel.trainable = True
#
# fineTuneAt = 100
#
# for layer in baseModel.layers[:fineTuneAt]:
#     layer.trainable = False
#
# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.RMSprop(learning_rate=baseLearningRate/10),
#               metrics=['accuracy'])
#
# fineTuneEpochs = 100
# totalEpochs = initialEpochs + fineTuneEpochs
#
# historyFine = model.fit(trainingDataset,
#                         epochs=totalEpochs,
#                         initial_epoch=history.epoch[-1],
#                         validation_data=validationDataset)
#
# acc += historyFine.history['accuracy']
# val_acc += historyFine.history['val_accuracy']
#
# loss += historyFine.history['loss']
# val_loss += historyFine.history['val_loss']
# #
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.ylim([0.3, 1])
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([0, 1.3])
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

# model.save("finished_model")

# savedModelDirectory = pathlib.Path('finished_model')
# converter = tf.lite.TFLiteConverter.from_saved_model('finished_model')
# tflite_model = converter.convert()
#
# with open("model.tflite", "wb") as f:
#     f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfliteModel = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tfliteModel)
