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


[//]: # (Image References)

[image1]: ./examples/architecture.png "Architecture"
[image2]: ./examples/Figure_1.png "Visualizing loss Epochs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  
---
I created a local environment in a MACOS Big Sur, Mid 2017. Please review the requirements in the file [environment.yml](https://github.com/rcgonzsv/Behavioral-Cloning-P3-rcgonzsv/blob/main/environment-plaidml-keras.yml). *Please note that you can skip the plaidml-keras step, since this was requirerd to run DL capabilities on my mac only*.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/rcgonzsv/Behavioral-Cloning-P3-rcgonzsv/blob/main/model.py) (script used to create and train the model)
* [drive.py](https://github.com/rcgonzsv/Behavioral-Cloning-P3-rcgonzsv/blob/main/drive.py)  (script to drive the car - feel free to modify this file)
* ['model.h5'](https://github.com/rcgonzsv/Behavioral-Cloning-P3-rcgonzsv/blob/main/models.model.h5) (a trained Keras model)
* a report [writeup template](https://github.com/rcgonzsv/Behavioral-Cloning-P3-rcgonzsv/blob/main/writeup_template.md) (either markdown or pdf)
* [video.mp4](https://github.com/rcgonzsv/Behavioral-Cloning-P3-rcgonzsv/blob/main/drive.py) (a video recording of your vehicle driving autonomously around the track for at least one full lap)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on **NVIDIA Autonomous vehicle team architecture**, that introduces a **normalization using Keras lambda layer** (code line 86 ) after **cropping** the orinal input from **3@160x320 to 3@66x200** (code line 83), please note in the code, that I also choosed a **pixel h/w/left/righ discrimination (74,20,60,60)** and explored/evaluated several scenarios regarding this and other preprocessing.

Also consist of **five convulotional layers, three with a minimun (0.1) dropout**, followed by **four fully connected layers** (see also architecture), and since the dimensions are changing between some of the convolutional layers, I was evaluating the trade of including/removing each of them comparing the loss for training and validation. The activation function I rely on is ReLu, for introducing nonlinearity (code lines 82 101 )

#### 2. Attempts to reduce overfitting in the model and data selection

As stated above, the model contains **3 dropout layer**s in order to **reduce overfitting** (model.py lines 89 91 93). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, however the first attempts took me a while since I tried different scenarios (own data set, new own data set including 2nd track, udacity data, udacity data augmented). I felt confortable using an augmented data from the original udacity data set, since I realized later on that other possible scenarios during the traing could be used (example: recovering from left deviation, obstacles, driving orientation, etc). I must say was quite challenging to learn to get the sense of how much data/time of training you would need , in order to prevent under/over fitting.  

I used this training data for training the model. I finally **randomly shuffled** the data set and put **20%** of the data into a **validation set**.The validation set helped determine if the model was over or under fitting. The **batch size was 32**, and the ideal number of epochs was **7**. I tried also with 3,4,5,10 having underfitting in (3,4), just enough with 5 (however the car drives off side of some points on the rodad) and not significative difference between 7 and 10 other that increasing computationa cost.

*Note: Fig 1.1 Visualized loss with 3 epochs*
![alt text][image2]


#### 3. Model parameter tuning

The model used an **[adam optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)**, so the learning rate was not tuned manually (model.py line 104). 

Also used mse since Mean Squared Error Loss, or MSE as preferred loss function, infering the maximum likelihood if the distribution of the targe variable is gaussian.Mean squared error is calculated as the average of the squared differences between the predicted and actual values. The result is always positive regardless of the sign of the predicted and actual values and a perfect value is 0.0. The squaring means that larger mistakes result in more error than smaller mistakes, meaning that the model is punished for making larger mistakes.

 

### Model Architecture 

![alt text][image1]

The models satisfied me in general terms, however eager to continue looking forward to improve and investigating similar approaches.




