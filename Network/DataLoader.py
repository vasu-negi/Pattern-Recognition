import pandas as pd
import numpy as np
import os
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils as npu
import keras
from keras.preprocessing.image import ImageDataGenerator

class DataLoader:
	def __init__(self):
		os.listdir('./')

	def data_loader(self):	
		raw_data = pd.read_csv('data/fer2013.csv')
		data = raw_data.values
		image_data = data[:, 1]
		values = data[:, 0]
		image_shape = image_data.shape[0]

		X = np.zeros((image_shape, 48*48))
		length_of_data = X.shape[0]
		features = X.shape[1]
		
		for i in range(length_of_data):
		    image = image_data[i].split(' ')
		    for j in range(features):
		        X[i, j] = int(image[j])

		data = X / 255
		num_train = 28710
		num_val = 32300

		categories = npu.to_categorical(values, 7)
		X_train = data[0:num_train, :]
		Y_train = categories[:num_train]

		X_val = data[num_train:num_val, :]
		Y_val = categories[num_train:num_val]
		
		X_train_shape = X_train.shape[0]
		X_val_shape = X_val.shape[0]

		X_train = X_train.reshape((X_train_shape, 48, 48,1 ))
		X_val = X_val.reshape((X_val_shape, 48, 48,1))
		
		data_generator = ImageDataGenerator(
		        rotation_range=10,  
		        zoom_range = 0.0,  
		        width_shift_range=0.1,  
		        height_shift_range=0.1)
		
		data_generator.fit(X_train)

		return [data_generator,X.shape[0],X_train,Y_train,X_val, Y_val]
	