#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Time : 2022/2/21 10:06
# @Author : Haozheng Li (Liam)
# @Email : hxl1119@case.edu

import utils
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam


"""
Global setting
Setting for using GPU, can be ignored
using CPU, could be a little bit slower
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_data():
    # Get data
    training_data = utils.read_csv_data(utils.TRAINING_FILE_PATH, is_training_data=True)
    testing_data = utils.read_csv_data(utils.TESTING_FILE_PATH)

    # data preprocessing
    """
    Note: In TensorFlow, the data needs to be converted into a 4-dimensional format when doing convolution
    Number of data, image height, image width, number of image channels
    """
    training_images = training_data['imagedata'].reshape(-1, 28, 28, 1)
    testing_images = testing_data['imagedata'].reshape(-1, 28, 28, 1)
    label = training_data['label']

    # Normalized
    training_images = training_images/255.0
    testing_images = testing_images/255.0

    # One-hot encoding
    label = tf.keras.utils.to_categorical(label, num_classes=10)

    # Build CNN model
    model = Sequential()

    # First layer：Convolution2D+polling
    first_layer_parameter = {'input_shape': (28, 28, 1),
                             'filters': 32,             # Number of filters
                             'kernel_size': 5,          # convolution kernel size
                             'strides': 1,
                             'padding': 'same',
                             'activation': 'relu'}
    model.add(Convolution2D(**first_layer_parameter))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',))

    # Second layer：Convolution2D+polling
    model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same'))

    model.add(Flatten())    # SOH

    # Third layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # fourth layer
    model.add(Dense(10, activation='softmax'))

    #  Compile and train
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(training_images, label, batch_size=64, epochs=10)

    # Get predict data
    predict_data = np.argmax(model.predict(testing_images), 1)
    print('CNN model final predict results are: {}'.format(predict_data))

    # Save data to csv
    utils.save_predict_data(utils.PATH_NEURAL_NETWORK, predict_data, 21000)

    # Show accuracy
    print('Model final training accuracy is {:.3f}%'.format(history.history['accuracy'][-1]*100))

    # Save model
    model.save('CNN_model.h5')
    
    # Draw figure
    draw_loss_figure(history)


def draw_loss_figure(history):
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['accuracy'], color='b')
    plt.title('model loss and acc')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'train_acc'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    process_data()
