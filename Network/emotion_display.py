import pandas as pd
import numpy as np
import os
import argparse
import cv2
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
import keras
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from matplotlib import pyplot as plt

from CVUtil import CVUtil
from DataLoader import DataLoader
from CreateNetwork import CreateNetwork

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



batch_size = 128
epochs = 1
dataloader = DataLoader()
datagen, steps_per_epoch, X_train, Y_train, X_val, Y_val = dataloader.data_loader()
steps_per_epoch = steps_per_epoch // batch_size


network_object = CreateNetwork()
model = network_object.create_model()

model.load_weights('model.h5')
video_capture = CVUtil()
video_capture.video_capture(model)

