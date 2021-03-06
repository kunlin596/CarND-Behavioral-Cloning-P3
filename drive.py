#!/usr/bin/env python3

import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

import tensorflow as tf

from model import TARGET_SPEED

# For issue `failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED`
# https://github.com/tensorflow/tensorflow/issues/45070
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
controller.set_desired(TARGET_SPEED)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        image_array = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        predictions = model.predict(image_array[None, :, :, :], batch_size=1)[0]

        steering_angle = float(predictions[0])
        # throttle = (float(predictions[1]) + 0.5)
        brake = float(predictions[2]) + 0.5
        speed = ((float(predictions[3]) + 0.5) * TARGET_SPEED) * (1 - brake)

        throttle = controller.update(speed)

        print('steering: {:10.8f}, throttle: {:5.3f}, brake: {:5.3f}, speed: {:5.3f}'.format(steering_angle, throttle, brake, speed))
        send_control(steering_angle, throttle, brake, speed)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0, 0)


def send_control(steering_angle, throttle, brake, speed):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle),
            # 'brake': str(brake),
            # 'speed': str(speed)
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    tf.compat.v1.reset_default_graph()
    with h5py.File(args.model, mode='r') as f:
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)

        model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
