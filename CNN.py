# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 09:38:51 2020

@author: ANKIT
"""

#Building the CNN-----> 1st part

#Importing the libraries
from keras.models import Sequential #initialise a neural network
from keras.layers import Conv2D #1st step to make CNN
from keras.layers import MaxPooling2D #2nd step Pooling step
from keras.layers import Dense #add fully connected layers
from keras.layers import Flatten #3rd step to convert pooling features into vector which becomes input of fully connected layers

#Initialise CNN
classifier = Sequential()

#Step 1- Convoluiton 
classifier.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step 2- Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2), ))

#Step 3- Flattening
classifier.add(Flatten())

#Step 4- Full Connected Steps
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the CNN----->2nd part
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/ANKIT/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/ANKIT/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=test_set,
        validation_steps=2000)

classifier.save('wallcrack_cnn_model.h5')