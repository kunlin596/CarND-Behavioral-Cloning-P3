# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

Description
---
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model
A [CNN By NVidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) is being implemented.

The overall strategy for deriving a model architecture was to input raw pixels into the network and output control signals like `steering angle` directly and feed it into the car controller and let it drive itself.
This network should focus on extracting the lane features, or anything that would appear on the sides of the roads frequently. Instead of manually detecting the lane like what we've done in the earlier projects, the goal here is detect the lane by itself and try to track it. The features on the road is not that complicated, and since the inputs are image, a CNN should do a decent job in this particular project.

#### Architecture summary
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

### Data samples

[Udacity Simulator](ttps://github.com/udacity/self-driving-car-sim) is used for data collection, below are 3 samples from the 3 cameras mounted on the car in the simulator.

| Left camera                                                  | Center camera                                                | Right camera                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/examples/left_2021_02_07_21_52_07_031.jpg) | ![](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/examples/center_2021_02_07_21_52_07_031.jpg) | ![](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/examples/right_2021_02_07_21_52_07_031.jpg) |

### Results
- Model: [track1.h5](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/track1.h5)
- Video: [run1.mp4](https://github.com/kunlin596/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4)
