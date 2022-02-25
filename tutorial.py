#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Time : 2022/2/23 20:59
# @Author : Haozheng Li (Liam)
# @Email : hxl1119@case.edu

# 3. Import libraries and modules

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Convolution2D, MaxPooling2D

from tensorflow.keras.datasets import mnist

np.random.seed(123)  # for reproducibility

def load_data(path='mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    #path = get_file(path,
    #                origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
    #                file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
(X_train, y_train), (X_test, y_test) = load_data()
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

import matplotlib.pyplot as plt

# plot 4 images as gray scale

plt.subplot(221)

print(y_train[4545],y_train[1],y_train[2],y_train[3])

plt.imshow(X_train[4545], cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

# show the plot

plt.show()



# Reshape the Input for the backend



X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)



plt.subplot(224)

plt.imshow(X_train[4545][0], cmap=plt.get_cmap('gray'))

plt.show()

# convert data type and normalize values

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255



print(y_train[4545])

plt.imshow(X_train[4545][0], cmap=plt.get_cmap('gray'))

plt.show()



print (y_train.shape)

Y_train = tf.one_hot(y_train, 10)

Y_test = tf.one_hot(y_test, 10)



print (Y_train.shape)

model = Sequential()

# add a sequential layer



# declare a input layer

model.add(Convolution2D(32,(3,3),activation='relu',data_format='channels_first',input_shape=(1,28,28)))



print (model.output_shape)



model.add(Convolution2D(32, 3, 3, activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))# output 10 classes corresponds to 0 to 9 digits we need to find



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

# model.load_weights('mn2.model')

model.fit(X_train, Y_train,batch_size=32, epochs=1, verbose=1)

model.save_weights('mn2.model')

score = model.evaluate(X_test, Y_test, verbose=0)

print(score)



k = np.array(X_train[4545]) #seven



print(k.shape)

y= k.reshape(1,1,28,28)

print(y.shape)



prediction = model.predict(y)

print(prediction)



class_pred = model.predict_classes(y)

print(class_pred)



plt.subplot(111)

plt.imshow(X_train[4545][0], cmap=plt.get_cmap('gray'))

plt.show()