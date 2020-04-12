import pandas as pd
import numpy as np
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
import keras
from keras.layers import Dense , Activation, Dropout,Flatten
from keras.constraints import maxnorm
from keras import backend as K
from keras.optimizers import SGD , Adam
from keras.layers import Conv2D , BatchNormalization,MaxPooling2D



class CreateNetwork:
	def create_model(self):
		model = Sequential()

		model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
		model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(7, activation='softmax'))

		print(model.summary())
		return model
