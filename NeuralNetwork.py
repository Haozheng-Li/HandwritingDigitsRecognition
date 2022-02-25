#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2022/2/20 15:11
# @Author : Haozheng Li (Liam)
# @Email : hxl1119@case.edu

import logging
import time

import utils
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from tensorflow.keras.utils import to_categorical


"""
Global setting
Setting for using GPU, can be ignored
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_data():
    # Get data
    training_data = utils.read_csv_data(utils.TRAINING_FILE_PATH, is_training_data=True)
    testing_data = utils.read_csv_data(utils.TESTING_FILE_PATH)
    
    # data preprocessing
    training_images = utils.data_preprocessing(training_data['imagedata'])
    testing_images = utils.data_preprocessing(testing_data['imagedata'])

    # Build neural network
    # Including two fully connected layer, contains 512 and 28 neurons respectively
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(utils.PIXEL_SIZE,)))
    model.add(Dense(10, activation='softmax'))
    adam = adam_v2.Adam(learning_rate=0.001)    # set eta

    # Compile and configure network
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(training_images, to_categorical(training_data['label']), epochs=5, batch_size=128)

    # Get predict data
    predict_data = np.argmax(model.predict(testing_images), 1)
    print('Neural network final predict results are: {}'.format(predict_data))

    # Save data to csv
    utils.save_predict_data(utils.PATH_NEURAL_NETWORK, predict_data, 21000)

    # Show accuracy
    print('Model final training accuracy is {:.3f}%'.format(history.history['accuracy'][-1]*100))

    # Save model
    model.save('NN_model.h5')

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
