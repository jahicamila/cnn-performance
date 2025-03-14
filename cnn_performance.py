from ast import Global
from multiprocessing import pool
from posixpath import split
import numpy as np
from numpy.lib.utils import source
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import sys
import warnings
import time
import csv
import fnmatch
import pydicom as dicom
import cv2
import keras_tuner as kt


def splitTrainDataByLabels():
    os.chdir('rsna-pneumonia-detection-challenge')
    if os.path.isdir('train/normal') is False:

        os.makedirs('train/normal')
        os.makedirs('train/pneumonia')

        def parseCsvRow(row):
            patientId = row[0]
            hasPneumonia = row[1] == 'Lung Opacity'
            isNormal = row[1] == 'Normal'
            imagePath = f'./train_images/size_256x256_INTER_LINEAR/{patientId}.jpeg'

            if os.path.isfile(imagePath):
                if hasPneumonia:
                    shutil.copy(imagePath, f'train/pneumonia/{patientId}.jpeg')
                if isNormal:
                    shutil.copy(imagePath, f'train/normal/{patientId}.jpeg')

        with open('./stage_2_detailed_class_info2.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    parseCsvRow(row)
                    
                line_count += 1

            print(f'Processed {line_count} lines.')

    os.chdir("..")

def splitDataToTrainTestValidation():
    os.chdir('rsna-pneumonia-detection-challenge')

    if os.path.isdir('test/normal') is False:

        os.makedirs('valid/normal')
        os.makedirs('valid/pneumonia')
        os.makedirs('test/normal')
        os.makedirs('test/pneumonia')

        def moveData(sourceDir, destinationDir, count):

            print(f'Move {count} images from {sourceDir} to {destinationDir}')

            for image in random.sample(glob.glob(f'{sourceDir}/*'), count):
                shutil.move(image, destinationDir)        

        def splitData(trainDir, validDir, testDir):
            splitCount = 1000
            moveData(trainDir, validDir, splitCount)
            moveData(trainDir, testDir, splitCount)

        splitData('train/normal', 'valid/normal', 'test/normal')
        splitData('train/pneumonia', 'valid/pneumonia', 'test/pneumonia')
    
    os.chdir("..")


def createCustomModel():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                            padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('relu'))

    model.add(Dense(2, activation='softmax'))

    return model


def createImageDataGenerator(train_path, valid_path, test_path):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_batches = train_datagen.flow_from_directory(
        directory=train_path,
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical'
    )
    valid_batches = test_datagen.flow_from_directory(
        directory=valid_path,
        target_size=(256, 256),
        batch_size=10,
        class_mode='categorical'
    )
    test_batches = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(256, 256),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    return train_batches, valid_batches, test_batches


def compileAndFitModel(train_batches, valid_batches, model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn = model.fit(x=train_batches,
              steps_per_epoch=len(train_batches),
              validation_data=valid_batches,
              validation_steps=len(valid_batches),
              epochs=50,
              verbose=2
    )
    return cnn


def plotTrainingAccuracyAndLoss(cnn):
    plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    plt.plot(cnn.history['loss'], color="blue", label = "Training Loss")
    plt.plot(cnn.history['val_loss'], color="orange", label = "Validation Loss")
    plt.title('Training and Validation Loss')
    plt.ylabel("Loss")
    plt.xlabel("Number of Epochs")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(cnn.history['accuracy'], color="green", label = "Training Accuracy")
    plt.plot(cnn.history['val_accuracy'], color="red", label = "Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Epochs")
    plt.legend()

    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def predictAndPlotConfusionMatrix(model, test_batches):

    predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

    cm_plot_labels = ['NORMAL','PNEUMONIA']
    
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

splitTrainDataByLabels()
splitDataToTrainTestValidation()

train_path = 'rsna-pneumonia-detection-challenge/train'
valid_path = 'rsna-pneumonia-detection-challenge/valid'
test_path = 'rsna-pneumonia-detection-challenge/test'


train_batches, valid_batches, test_batches = createImageDataGenerator(train_path, valid_path, test_path)

model = createCustomModel()
model.summary()

cnn = compileAndFitModel(train_batches, valid_batches, model)
plotTrainingAccuracyAndLoss(cnn)
predictAndPlotConfusionMatrix(model, test_batches)
