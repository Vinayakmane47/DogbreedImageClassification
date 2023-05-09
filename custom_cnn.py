import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import pathlib
from keras.callbacks import Callback
import math
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout, Flatten,Activation, BatchNormalization,MaxPooling2D
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator


## Data Augmentation : 
# Define the directory containing the images
data_dir = "/kaggle/input/dogs-breed-dataset/dog_v1" ## Enter your folder location . 

# Define the parameters for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    rescale=1./255
)

# Define the training and validation generators
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


## Prepare Model 
model = Sequential()

# Add convolutional layers
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
# Add fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))


### Warming up the Learning rate 


class LearningRateWarmupCallback(Callback):
    def __init__(self, warmup_batches, init_lr, verbose=0):
        super(LearningRateWarmupCallback, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose

    def on_train_batch_begin(self, batch, logs=None):
        if batch < self.warmup_batches:
            lr = self.init_lr * (batch + 1) / self.warmup_batches
            if self.verbose > 0:
                print(f'Batch {batch}: Learning rate = {lr}')
            self.model.optimizer.lr = lr
            
 



# Define early stopping callback
early_stop = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)


history = model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define the number of epochs to train the model
epochs = 50

# Define the warmup period and initial learning rate
warmup_batches = 5
init_lr = 0.001

# Define the learning rate warmup callback
lr_callback = LearningRateWarmupCallback(warmup_batches, init_lr)

# Train the model using the fit method
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop]
)            
