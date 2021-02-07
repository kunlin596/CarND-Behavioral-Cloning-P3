#!/usr/bin/env python3

import os
import cv2
import collections

import numpy as np
import pandas as pd
import seaborn as sns  # noqa

from IPython import embed
import tensorflow as tf

# For issue `failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED`
# https://github.com/tensorflow/tensorflow/issues/45070
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


DATA_FOLDER_NAME = 'carnd-pj3-data'
DATA_PATH = os.path.join(os.environ['HOME'], 'dev/data/', DATA_FOLDER_NAME)
CSV_PATH = os.path.join(DATA_PATH, 'driving_log.csv')


np.set_printoptions(precision=3, suppress=True)


def read_data(csvfile):
    column_names = collections.OrderedDict({
        'center': 0,
        'left': 1,
        'right': 2,
        'steering': 3,
        'throttle': 4,
        'brake': 5,
        'speed': 6
    })

    print('Start reading data')
    raw_dataset = pd.read_csv(csvfile,
                              header=0,
                              names=column_names.keys(),
                              sep=',',
                              skipinitialspace=True)
    print('End reading data')

    for row in raw_dataset.to_numpy():
        image_center = cv2.imread(os.path.join(DATA_PATH, row[column_names['center']]))
        if image_center is not None:
            image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)

        image_left = cv2.imread(os.path.join(DATA_PATH, row[column_names['left']]))
        # if image_left is not None:
        #     image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

        image_right = cv2.imread(os.path.join(DATA_PATH, row[column_names['right']]))
        # if image_right is not None:
        #     image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

        measurement = float(row[column_names['steering']])
        yield image_left, image_center, image_right, measurement
        yield image_left, np.fliplr(image_center), image_right, -measurement  # data augmentation


def get_model(X_train, y_train):
    from keras.models import Sequential
    from keras import layers

    model = Sequential([
        # Preprocessing, RGB to gray scale
        layers.Conv2D(1, (1, 1), activation='relu', data_format="channels_last", input_shape=(160, 320, 3)),
        # Normalize image to be 0-means and range from 0 to 1
        layers.Lambda(lambda x: x / 255.0 - 0.5),
        # LeNet
        # Convolutional layers
        layers.Conv2D(6, (5, 5), activation='relu'),
        layers.Dropout(0.2),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPool2D((2, 2)),
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(1)
    ])

    embed()

    model.compile(
        loss='mse',
        optimizer='adam'
    )

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
    model.save('model.h5')
    tf.compat.v1.reset_default_graph()


def main():
    images_center = []
    measurements = []
    for data in read_data(CSV_PATH):
        images_center.append(data[1])
        measurements.append(data[-1])

    X_train = np.array(images_center)
    y_train = np.array(measurements)
    get_model(X_train, y_train)


if __name__ == '__main__':
    main()
