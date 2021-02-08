# **Behavioral Cloning**

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py track1.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model and Training
As hinted in the lectures, both `LeNet` and a [CNN By NVidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) is being tested.

#### 1. An appropriate model architecture overview
##### 1.1 Preprocessing

The data is read in gray scale image in order to remove influence by the color (since in this particular project, we are mostly expecting the network to learn the lane features)

__Data Augmentation__
From `model.py: L97`, I constructed the data in this way. For every record in the CSV file, I will give 6 training samples. The first 3 are the raw data captured from the camera, the later 3 samples are data augmentations for compensating the directional (operational) bias for the track. This already reached the goal of this project fairly well so I didn't employ further data augmentation processes. One may consider manually rotate or translate the sample a bit to create more artificial samples.

Here, `steering_correction = 0.15`
```
yield image_left              , steering + steering_correction  , throttle, brake, speed  # noqa: E203
yield image_center            , steering                        , throttle, brake, speed  # noqa: E203
yield image_right             , steering - steering_correction  , throttle, brake, speed  # noqa: E203
yield np.fliplr(image_left)   , -steering - steering_correction , throttle, brake, speed  # noqa: E203
yield np.fliplr(image_center) , -steering                       , throttle, brake, speed  # noqa: E203
yield np.fliplr(image_right)  , -steering + steering_correction , throttle, brake, speed  # noqa: E203
```
__Data Normalization__
The image is normalized to be by `image = image / 255.0 - 0.5` to be zero-meaned.

The same normalizations are also being done the predictions values like `speed` as well, I observed that in the simulator, the maximum speed of the car is around `32`, so I normalized the `speed` using `speed = speed / TARGET_SPEED - 0.5`. Since it looks like that `brake` and `throttle` are already normalized in `[0, 1]`, so I simply subtracted 0.5 from them.

__Data cropping__
In the image, nearly top half of the image was not from the road (our ROI), and the bottom half is coming from the car itself, so a cropping layer is added to the network with some empirical crop margins.
```
layers.Cropping2D(cropping=((50, 20), (0, 0)), data_format="channels_last", input_shape=image_shape),
```

##### 1.2 Network architecture
The code below the actual model based on the NVidia CNN architecture with some additional layers to avoid over fitting.
Note that since I've converted the input image to be gray scale, the input size is `160 x 320 x1`.

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d (Cropping2D)      (None, 90, 320, 1)        0
_________________________________________________________________
lambda (Lambda)              (None, 90, 320, 1)        0
_________________________________________________________________
conv2d (Conv2D)              (None, 86, 316, 24)       624
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 43, 158, 24)       0
_________________________________________________________________
dropout (Dropout)            (None, 43, 158, 24)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 39, 154, 36)       21636
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 19, 77, 36)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 19, 77, 36)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 75, 48)        15600
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 37, 48)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 37, 48)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 3, 17, 64)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 17, 64)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 15, 64)         36928
_________________________________________________________________
flatten (Flatten)            (None, 960)               0
_________________________________________________________________
dense (Dense)                (None, 1164)              1118604
_________________________________________________________________
dense_1 (Dense)              (None, 100)               116500
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 4)                 44
=================================================================
Total params: 1,343,208
Trainable params: 1,343,208
Non-trainable params: 0
_________________________________________________________________
```

##### 1.3 Outputs and their usages
There are 4 outputs here,
* steering
* throttle
* brake
* speed

`speed` is normalized to be in the same range as angle `[-1, 1]` (maximum speed of the car in the simulator is roughly 30.19, so the range of actual speed is considered to be `[0, 32]`).

In `drive.py`, the four predictions are used in this way,
```
# TARGET_SPEED = 32
steering_angle = float(predictions[0])
# throttle = (float(predictions[1]) + 0.5)
brake = float(predictions[2]) + 0.5
speed = ((float(predictions[3]) + 0.5) * TARGET_SPEED) * (1 - brake)
```
Note that, I'm not sure on how to use `throttle` so it's not used.

#### 2. Attempts to reduce over fitting in the model

In addition to the structure presented in the paper mentioned above by NVidia, after each max pooling layer, additional drop out layers are added in order to avoid over fitting.
Also since the training process converged quickly, only 3 epochs are used for training.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Also the steering angle fix for images from left and right are set to be 0.15. This angle fix should really computed using the some camera calibration information and trigonometry to compute, but for the sake of simplicity, it's set to be `0.15`.

#### 4. Training data

My training dataset includes:
1. The original data samples provided in the course.
2. 2 loops of focusing on stay in the middle of the lane.
3. 1 loops of focusing on doing zig-zag movements in order for the model to capture how to move back to the center.
4. 1 loop of clock-wise (opposed to default starting position) driving.

#### 5. Traing

The overall strategy for deriving a model architecture was to input raw pixels into the network and output control signals like `steering angle` directly and feed it into the car controller and let it drive itself.
I think network should focus on extracting the lane features, or anything which could appear on the sides of the roads frequently. Instead of manually detecting the lane like what we've done in the earlier projects, the goal here is detect the lane by itself and try to track it. The features on the road is not that complicated, and since the inputs are image, a CNN should do a decent job in this particular project.

Some key points about the training process:
1. All 4 values collected from the simulator are being predicted.
2. All images from 3 cameras mounted on the car are used as input (and w/ more data augmentation).
3. Data samples are spitted into 8:2 for training and validation.
4. Generators and batches are used for training to keep GPU memory being overwhelmed.
5. `steering angle`, `speed` and `brake` from the model predictions are used to control the car.
6. Adam optimizer is used for automatically setting learning rate.

#### 5. Result
As a final result, the car can drive at full speed in the simulator for consecutive multiple loops w/o any issues.

- Model: [track1.h5](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/track1.h5)
- Video: [run1.mp4](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4)
