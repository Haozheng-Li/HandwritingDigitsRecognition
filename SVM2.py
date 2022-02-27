#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Time : 2022/2/20 21:33
# @Author : Haozheng Li (Liam)
# @Email : hxl1119@case.edu


from sklearn import svm
import numpy as np
import utils


def process_data():
    # Get data
    training_data = utils.read_csv_data(utils.TRAINING_FILE_PATH, True)
    testing_data = utils.read_csv_data(utils.TESTING_FILE_PATH)

    training_images = training_data['imagedata']
    label = training_data['label']
    testing_images = testing_data['imagedata']

    # Build SVM model
    model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')

    # Training
    history = model.fit(training_images, label)

    # Get predict data
    # It may take some time
    predict_data = model.predict(testing_images)
    print('SVM final predict results are: {}'.format(predict_data))

    # Save data to csv
    utils.save_predict_data(utils.PATH_NEURAL_NETWORK, predict_data, 21000)

    # Show accuracy
    test_result = model.predict(training_images)
    print('Model final training accuracy is {:.3f}%'.format(np.sum(np.equal(test_result, label)) / 21000 * 100))


if __name__ == '__main__':
    process_data()


