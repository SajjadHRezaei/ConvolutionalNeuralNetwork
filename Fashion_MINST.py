import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.callbacks import ModelCheckpoint

!rm -rf ./logs/ 

%load_ext tensorboard

