#!/usr/bin/env python3

import os
import cv2
import collections
import math
import argparse
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as K

import seaborn as sns  # noqa: F401
import sklearn  # noqa: F401
from IPython import embed  # noqa: F401


# For issue `failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED`
# https://github.com/tensorflow/tensorflow/issues/45070
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

np.set_printoptions(precision=3, suppress=True)

TARGET_SPEED = 31.0
COLUMN_NAMES = collections.OrderedDict({
    'center': 0,
    'left': 1,
    'right': 2,
    'steering': 3,
    'throttle': 4,
    'brake': 5,
    'speed': 6
})


class DataGenerator(K.utils.Sequence):

    def __init__(self, samples, batch_size=4, dim=(160, 320, 1)):
        self._batch_size = batch_size
        self._samples = samples
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
            for data in self._read_data(sample):
                image, steering, throttle, brake, speed = data
                images.append(image)
                outputs = [
                    steering,
                    throttle,
                    brake,
                    speed
                ]
                measurements.append(outputs)

        return np.asarray(images), np.asarray(measurements)

    def on_epoch_end(self):
        np.random.shuffle(self._indices)

    @staticmethod
    def _read_data(sample, steering_correction=0.15):
        # print('reaing %s' % sample[COLUMN_NAMES['center']])
        image_center = cv2.imread(sample[COLUMN_NAMES['center']])
        image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

        image_left = cv2.imread(sample[COLUMN_NAMES['left']])
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

        image_right = cv2.imread(sample[COLUMN_NAMES['right']])
        image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)[..., np.newaxis]

        steering = float(sample[COLUMN_NAMES['steering']])
        throttle = float(sample[COLUMN_NAMES['throttle']])
        brake = float(sample[COLUMN_NAMES['brake']])
        speed = float(sample[COLUMN_NAMES['speed']])

        yield image_left              , steering + steering_correction  , throttle, brake, speed  # noqa: E203
        yield image_center            , steering                        , throttle, brake, speed  # noqa: E203
        yield image_right             , steering - steering_correction  , throttle, brake, speed  # noqa: E203
        yield np.fliplr(image_left)   , -steering - steering_correction , throttle, brake, speed  # noqa: E203
        yield np.fliplr(image_center) , -steering                       , throttle, brake, speed  # noqa: E203
        yield np.fliplr(image_right)  , -steering + steering_correction , throttle, brake, speed  # noqa: E203


def _read_data_file(csvfile):
    print('--- Reading csvfile %s ---' % csvfile)
    samples = pd.read_csv(csvfile, header=0, names=COLUMN_NAMES.keys(),
                          sep=',', skipinitialspace=True)

    records = samples.to_numpy()
    dirname = os.path.dirname(csvfile)
    print('Found %d records' % len(records))
    for index, row in enumerate(records):
        for i in [COLUMN_NAMES['left'], COLUMN_NAMES['center'], COLUMN_NAMES['right']]:
            if not os.path.exists(records[index][i]) or not os.path.isfile(records[index][i]):
                records[index][i] = os.path.join(dirname, records[index][i])
        records[index][COLUMN_NAMES['steering']] = float(records[index][COLUMN_NAMES['steering']])
        records[index][COLUMN_NAMES['throttle']] = float(records[index][COLUMN_NAMES['throttle']]) - 0.5
        records[index][COLUMN_NAMES['brake']] = float(records[index][COLUMN_NAMES['brake']]) - 0.5
        records[index][COLUMN_NAMES['speed']] = float(records[index][COLUMN_NAMES['speed']]) / TARGET_SPEED - 0.5

    return records


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
    model.add(layers.Dropout(0.2))
    model.add(layers.Convolution2D(36, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Convolution2D(48, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1164, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    return model


def get_model(image_shape, output_shape=1, modeltype='LeNet'):
    from keras.models import Sequential
    from keras import layers

    model = Sequential([
        # Crop top region of the image, top, bottom, left, right
        layers.Cropping2D(cropping=((50, 20), (0, 0)), data_format="channels_last", input_shape=image_shape),
        # Normalize image to be 0-meaned and range from 0 to 1
        layers.Lambda(lambda x: x / 255.0 - 0.5),
    ])

    if modeltype == 'LeNet':
        model = _get_LeNet(model)
    elif modeltype == 'NVidia':
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


def train_model(modelname, train_generator, validation_generator, model):
    print('-' * 80)
    print('Train model')
    print('-' * 80)
    # FIXME: Use Model.fit, already support generator.
    history_object = model.fit_generator(generator=train_generator,
                                         validation_data=validation_generator,
                                         epochs=3,
                                         verbose=1,
                                         workers=16,
                                         shuffle=True)
    model.save('%s.h5' % modelname)
    tf.compat.v1.reset_default_graph()
    return history_object


def main(modelname, datafolder):
    allsamples = None
    for folder in datafolder:
        datapath = os.path.join(os.environ['HOME'], 'dev/data/', folder)
        csvpath = os.path.join(datapath, 'driving_log.csv')
        samples = _read_data_file(csvpath)
        if allsamples is None:
            allsamples = samples
        else:
            allsamples = np.vstack([allsamples, samples])

    print('--- Read %d samples ---' % len(allsamples))
    train_samples, validation_samples = train_test_split(allsamples, test_size=0.2)
    modeltype = 'NVidia'
    model = get_model(image_shape=(160, 320, 1), output_shape=4, modeltype=modeltype)
    print('--- Created %s model ---' % modeltype)

    history_object = train_model(modelname,
                                 DataGenerator(train_samples),
                                 DataGenerator(validation_samples),
                                 model)

    show_history(history_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--modelname', '-m', type=str, default='model', help='Model name to be saved.')
    parser.add_argument('--datafolder', '-d', type=str, nargs='+', default=None, help='Path to the data folder.')
    args = parser.parse_args()

    main(args.modelname, args.datafolder)
