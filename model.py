import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Cropping2D, Lambda, Dropout
from keras.layers.pooling import MaxPooling2D



def loadData(col, images, measurement, steering_measurements):
    current_path = image_path + '/' + col.strip()

    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(np.asarray(image))
    steering_measurements.append(measurement)

    # Flip randomly to data augmentation
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        image_flipped = np.fliplr(image)
        images.append(np.asarray(image_flipped))
        measurement_flipped = measurement * (-1)
        steering_measurements.append(measurement)


def getImgMes(sample):
    images = []
    steering_measurements = []
    for line in sample[0:]:
        measurement = float(line[3])
        # Get random adjusted data steering measurements for all sides camera images
        correction = 0.25
        camera = np.random.choice(['center', 'left', 'right'])
        if camera == 'center':
            col_center = line[0]
            loadData(col_center, images, measurement, steering_measurements)
        elif camera == 'left':
            col_left = line[1]
            loadData(col_left, images, measurement + correction, steering_measurements)
        else:
            col_right = line[2]
            loadData(col_right, images, measurement - correction, steering_measurements)
    return images, steering_measurements

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for image, measurement in batch_samples:
                images.append(image)
                measurements.append(measurement)
            x_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(x_train, y_train)

if __name__ == '__main__':

	# Define path, load data, define, compile, train and save the model
	image_path = 'data'
	driving_log_path = 'data/driving_log.csv'

	rows = []
	with open(driving_log_path) as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        rows.append(row)

	X_total, y_total = getImgMes(rows[1:])

	model = Sequential()

    # CNN Architecture
	model.add(Cropping2D(cropping = ((74,20), (60,60)),input_shape=(160, 320, 3))) # Input planes, set cropping to 3@66x200
    #model.add(Cropping2D(cropping = ((70,25), (0,0)),input_shape=(160, 320, 3))) # Udacity lesson cropping suggestion

	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3))) # Normalized input planes 3@66x200 lambda setup
    # Kernel 5x5
	model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu')) # Convolutional feature map 24@31x98
	model.add(Dropout(.1)) # Dropout
	model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu')) # Convolutional feature map 36@14x47
	model.add(Dropout(.1)) # Dropout
	model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu')) # Convolutional feature map 48@5x22
	model.add(Dropout(.1)) # Dropout
	model.add(Conv2D(64, 3, 3, activation='relu')) # Convolutional feature map 64@3x20
	model.add(Conv2D(64, 3, 3, activation='relu')) # Convolutional feature map 64@1x18

	model.add(Flatten()) # Flatten 1164 neurons
	model.add(Dense(100)) # Fully-connected layer, 100 neurons
	model.add(Dense(50)) # Fully-connected layer, 50 neurons
	model.add(Dense(10)) # Fully-connected layer, 10 neurons
	model.add(Dense(1)) # Output vehicle control

    # compile and train the model using the generator function
	model.compile(loss='mse', optimizer='adam')


	print('Training model')
	samples = list(zip(X_total, y_total))
	train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
	train_generator = generator(train_samples, batch_size = 32) # Set our batch size for training
	validation_generator = generator(validation_samples, batch_size = 32) # Set our batch size for validation

	history_object = model.fit_generator(train_generator,
	                                    samples_per_epoch = len(train_samples),
	                                    validation_data = validation_generator,
	                                    nb_val_samples = len(validation_samples),
	                                    nb_epoch = 3,
	                                    verbose = 1)
	print('Training Completed...')



# Saving the model
print()
print('Please wait your model is being saved')
model.save('models/model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
