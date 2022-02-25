#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @Time : 2022/2/20 19:30
# @Author : Haozheng Li (Liam)
# @Email : hxl1119@case.edu
import os.path
import time
import logging
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESOLUTION = (28, 28)
PIXEL_SIZE = 28 * 28

PATH_NEURAL_NETWORK = 'NN_results.csv'
PATH_CNN = 'CNN_results.csv'
PATH_SVM = 'SVM_results.csv'

TRAINING_FILE_PATH = 'Data/training.csv'
TESTING_FILE_PATH = 'Data/testing.csv'

logging.basicConfig(level=logging.INFO)


def log(func):
    def inner(*args, **kwargs):
        begin_time = time.time()
        logging.debug(' Begin func: {}'.format(func.__name__))
        result = func(*args, **kwargs)
        logging.debug(' End func: {}, time consuming: {}s'.format(func.__name__, time.time()-begin_time))
        return result
    return inner


def save_predict_data(filename, results, id_range):
    """
    Save predict data to csv
    :param id_range: int
    :param filename: str
    :param results: list
    :return: None
    """
    df = pandas.DataFrame({'id': range(id_range), 'results': results})
    df.to_csv(os.getcwd() + '/Data/' + filename, index=False, sep=',')


@log
def read_csv_data(file_path, is_training_data=False):
    """
    :param file_path: str
    :param is_training_data: bool
    :return: dict {'imagedata': np.array, 'label': np.array}
    """
    logging.info("Reading {} data".format(file_path))
    data = {'imagedata': [],
            'label': []}
    f = open(file_path, 'r')
    for line in f.readlines()[1:]:
        datalist = line.strip().split(',')
        if is_training_data:
            data['label'].append(int(datalist[0]))
            data['imagedata'].append([int(column) for column in datalist[1:] if column.isdigit()])
        else:
            data['imagedata'].append([int(column) for column in datalist if column.isdigit()])
    data['imagedata'] = np.array(data['imagedata'])
    data['label'] = np.array(data['label'])
    logging.debug('Data dimension: {}'.format(data['imagedata'].shape))
    logging.debug('Numbers of figure: {}'.format(len(data['imagedata'])))
    return data


def read_predict_data(filename):
    """
    Read test data and show its figure
    :param filename: one of RESULTS_FILE_NAME
    :return:
    """
    index = int(input("Input an id, get predicted result and figure: "))
    alldata = pd.read_csv('Data/' + filename)
    try:
        results = alldata.iloc[index]
        print("The predicted result is: {}".format(results['results']))
        image_data = read_csv_data(TESTING_FILE_PATH)['imagedata'][index]
        plt.imshow(image_data.reshape(28, 28))
        plt.show()
    except IndexError:
        print('Index input error, please try again')


def data_preprocessing(data):
    data = data.reshape((21000, PIXEL_SIZE))
    return data.astype('float32') / 255


def show_figure(data):
    plt.axis('off')
    plt.imshow(data.reshape(*RESOLUTION))
    plt.show()


if __name__ == '__main__':
    read_predict_data(PATH_NEURAL_NETWORK)
