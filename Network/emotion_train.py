import pandas as pd
import numpy as np
import os
import argparse
import cv2
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
import keras
from keras.layers import Dense , Activation,Dropout,Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , Adam
from keras.layers import Conv2D , BatchNormalization,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from matplotlib import pyplot as plt

from CVUtil import CVUtil
from DataLoader import DataLoader
from CreateNetwork import CreateNetwork
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
a = ap.parse_args()
mode = a.mode


batch_size = 128
epochs = 1
dataloader = DataLoader()
datagen,steps_per_epoch,X_train,Y_train,X_val,Y_val = dataloader.data_loader()
steps_per_epoch = steps_per_epoch // batch_size


network_object = CreateNetwork()
model = network_object.create_model()


model.compile(loss='categorical_crossentropy',
	optimizer='adam' ,
	metrics=['accuracy'])


history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch),
					steps_per_epoch=steps_per_epoch,
					validation_data =(X_val,Y_val),
					epochs = epochs,
					verbose = 2)

model.save_weights('model.h5')

#history.history is a dictionary which stores the values for the accuracy (['val_loss', 'val_accuracy', 'loss', 'accuracy'])

history_dict = history.history


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


