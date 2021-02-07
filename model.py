#!/usr/bin/env python3

import os
import cv2
import collections
import math
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as K
from drive import TARGET_SPEED

import seaborn as sns  # noqa: F401
import sklearn  # noqa: F401
from IPython import embed  # noqa: F401


# For issue `failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED`
# https://github.com/tensorflow/tensorflow/issues/45070
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


DATA_FOLDER_NAME = 'carnd-pj3-data'
DATA_PATH = os.path.join(os.environ['HOME'], 'dev/data/', DATA_FOLDER_NAME)
CSV_PATH = os.path.join(DATA_PATH, 'driving_log.csv')


np.set_printoptions(precision=3, suppress=True)


class DataGenerator(K.utils.Sequence):

    def __init__(self, samples, column_names=None, batch_size=1, dim=(160, 320, 1)):
        self._batch_size = batch_size
        self._samples = samples.to_numpy()
        self._column_names = column_names
        self._indices = np.arange(samples.shape[0])

    def __len__(self):
        # must implement
        return math.ceil(len(self._samples) / self._batch_size)

    def __getitem__(self, index):
        # must implement
        indices = self._indices[index * self._batch_size: (index + 1) * self._batch_size]
        batch_samples = self._samples[indices]

        images = []
        measurements = []

        for sample in batch_samples:
            for data in self._read_data(sample, self._column_names):
                image, steering, throttle, brake, speed = data
                images.append(image)
                outputs = [
                    steering,
                    throttle * 2 - 0.5,
                    brake * 2 - 0.5,
                    speed / TARGET_SPEED
                ]
                measurements.append(outputs)

        return np.asarray(images), np.asarray(measurements)

    def on_epoch_end(self):
        np.random.shuffle(self._indices)

    @staticmethod
    def _read_data(sample, column_names, steering_correction=0.2):
        image_center = cv2.imread(os.path.join(DATA_PATH, sample[column_names['center']]))
        if image_center is not None:
            image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

        image_left = cv2.imread(os.path.join(DATA_PATH, sample[column_names['left']]))
        if image_left is not None:
            image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

        image_right = cv2.imread(os.path.join(DATA_PATH, sample[column_names['right']]))
        if image_right is not None:
            image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

        steering = float(sample[column_names['steering']])
        throttle = float(sample[column_names['throttle']])
        brake = float(sample[column_names['brake']])
        speed = float(sample[column_names['speed']])

        yield image_left              , steering + steering_correction  , throttle, brake, speed  # noqa: E203
        yield image_center            , steering                        , throttle, brake, speed  # noqa: E203
        yield image_right             , steering - steering_correction  , throttle, brake, speed  # noqa: E203
        yield np.fliplr(image_left)   , -steering - steering_correction , throttle, brake, speed  # noqa: E203
        yield np.fliplr(image_center) , -steering                       , throttle, brake, speed  # noqa: E203
        yield np.fliplr(image_right)  , -steering + steering_correction , throttle, brake, speed  # noqa: E203


def read_data_file(csvfile):
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
    samples = pd.read_csv(csvfile, header=0, names=column_names.keys(),
                          sep=',', skipinitialspace=True)
    return samples, column_names


def _get_LeNet(model):
    from keras import layers
    model.add(layers.Conv2D(6, (5, 5), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))


def _get_NVidia(model):
    # https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    from keras import layers
    model.add(layers.Convolution2D(24, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(36, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(48, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    return model


def get_model(image_shape, output_shape=1, model_name='LeNet'):
    from keras.models import Sequential
    from keras import layers

    model = Sequential([
        # Preprocessing, RGB to gray scale
        # Crop top region of the image, top, bottom, left, right
        layers.Cropping2D(cropping=((50, 20), (0, 0)), data_format="channels_last", input_shape=image_shape),
        # Normalize image to be 0-meaned and range from 0 to 1
        layers.Lambda(lambda x: x / 255.0 - 0.5),
    ])

    if model_name == 'LeNet':
        model = _get_LeNet(model)
    elif model_name == 'NVidia':
        model = _get_NVidia(model)

    # Output layer
    model.add(layers.Dense(output_shape))

    model.compile(
        loss='mse',
        optimizer='adam'
    )

    return model


def show_history(history_object):
    import matplotlib.pyplot as plt
    print(history_object.history.keys())
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show(block=False)
    plt.ion()
    print('-' * 80)
    print('Traning history')
    print('-' * 80)
    embed()


def train_model(train_generator, validation_generator, model):
    print('-' * 80)
    print('Train model')
    print('-' * 80)
    history_object = model.fit_generator(generator=train_generator,
                                         validation_data=validation_generator,
                                         epochs=3,
                                         verbose=1,
                                         workers=16,
                                         shuffle=True)
    model.save('model.h5')
    tf.compat.v1.reset_default_graph()
    return history_object


def main():
    samples, column_names = read_data_file(CSV_PATH)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    model = get_model(image_shape=(160, 320, 1), output_shape=4, model_name='NVidia')
    history_object = train_model(DataGenerator(train_samples, column_names),
                                 DataGenerator(validation_samples, column_names),
                                 model)

    show_history(history_object)


if __name__ == '__main__':
    main()
