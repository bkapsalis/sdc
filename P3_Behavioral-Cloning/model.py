import os
import csv
import cv2
import numpy as np

import sklearn
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
#from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D 
np.random.seed(23)

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader,None)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.05, random_state = 23)

print(len(samples))
def generator(samples, batch_size=32):
    num_samples = (len(samples))
    print(num_samples)
    
    while 1: # Loop forever so the generator never terminates
        #np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '../data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                  
                correction = 0.2    
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)
                
            augmented_images = []
            augmented_angles = []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                flipped_image = cv2.flip(image, 1)
                flipped_angle = float(angle) * -1.0
                augmented_images.append(flipped_image)
                augmented_angles.append(flipped_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(X_train, y_train)
            
            yield sklearn.utils.shuffle(X_train, y_train, random_state = 23)
            #return sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=15)
validation_generator = generator(validation_samples, batch_size=15)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

model.add(Lambda(lambda x: x/ 255.0 - 0.5, input_shape= (160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Convolution2D(24,5, 5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3, 3, activation='relu'))
model.add(Convolution2D(64,3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=2)
model.save('model.h5')