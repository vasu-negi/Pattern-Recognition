import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

		return model
