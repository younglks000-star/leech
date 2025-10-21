# MT-IceNet — Notebook export
# Converted to Spyder-friendly script with cell separators (`# %%`).
# Source: MT_IceNet.ipynb
# NOTE: Markdown cells are preserved as comments.

# %% [code cell 1]
import os
import math
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM,TimeDistributed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# %% [code cell 2]
#Enabling Dynamic Memory Allocation
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# %% [markdown]
# ## Sea Ice Prediction - MT-IceNet
# 
# ### Loading Rolling Window Data 1979-2021
# 
# 
# #### Features:
# longwave, rain_rate, snow_rate, sst, sea_ice

# %% [code cell 3]
x_train = np.load("data/X_train_rolling_filled_5f.npy")
y_train = np.load("data/y_train_rolling_filled_final.npy")
x_test = np.load("data/X_test_rolling_filled_5f.npy")
y_test = np.load("data/y_test_rolling_filled_final.npy")
land_mask = np.load("data/y_land_mask_actual.npy",allow_pickle=True)

# %% [code cell 4]
x_train2 = np.load("data/X_train_convlstm_rolling_biweekly_5f.npy")
x_test2 = np.load("data/X_test_convlstm_rolling_biweekly_5f.npy")

# %% [code cell 5]
print('x_train.shape:',x_train.shape)
print('x_train2.shape:',x_train2.shape)
print('y_train.shape:',y_train.shape)

print('x_test.shape:',x_test.shape)
print('x_test2.shape:',x_test2.shape)
print('y_test.shape:',y_test.shape)

# %% [code cell 6]
lag = 6
x_train = x_train[:-lag,:,:,:,:]
x_test = x_test[:-lag,:,:,:,:]

x_train2 = x_train2[:-(lag+1),:,:,:,:]
x_test2 = x_test2[:-(lag+1),:,:,:,:]

y_train = y_train[lag:,:,:,:]
y_test = y_test[lag:,:,:,:]

# %% [code cell 7]
print('x_train.shape:',x_train.shape)
print('x_train2.shape:',x_train2.shape)
print('y_train.shape:',y_train.shape)

print('x_test.shape:',x_test.shape)
print('x_test2.shape:',x_test2.shape)
print('y_test.shape:',y_test.shape)

# %% [code cell 8]
#Replacing all nans with Zeros
x_train = np.nan_to_num(x_train)
x_train2 = np.nan_to_num(x_train2)
y_train = np.nan_to_num(y_train)
x_test = np.nan_to_num(x_test)
x_test2 = np.nan_to_num(x_test2)
y_test = np.nan_to_num(y_test)

# %% [markdown]
# ### Reshaping Input and Target Features

# %% [code cell 9]
# convert an array of values into a dataset matrix
def reshape_features(dataset, samples, timestep, lat, lon, features):
    print(dataset.shape)
    X = dataset.reshape(samples, timestep, lat, lon, features)
    return X

# convert an array of values into a dataset matrix
def reshape_outcome(dataset, months, lat, lon):
    print(dataset.shape)
    X = dataset.reshape(months, lat, lon, 1)
    return X

# %% [markdown]
# ### Normalization

# %% [code cell 10]
NUM_TRAIN = len(x_train)
NUM_TEST = len(x_test)
print(NUM_TRAIN)
print(NUM_TEST)

# %% [code cell 11]
# normalize the features

scaler_a = StandardScaler()
#x_train = scaler_a.fit_transform(x_train.reshape(-1,x_train.shape[3])) #reshaping to 2d for standard scaling
#x_test = scaler_a.transform(x_test.reshape(-1,x_test.shape[3])) #reshaping to 2d for standard scaling

x_train = scaler_a.fit_transform(x_train.reshape(-1,1)) #reshaping to 2d for standard scaling
x_test = scaler_a.transform(x_test.reshape(-1,1)) #reshaping to 2d for standard scaling

scaler_b = StandardScaler()
#x_train2 = scaler_b.fit_transform(x_train2.reshape(-1,x_train2.shape[3])) #reshaping to 2d for standard scaling
#x_test2 = scaler_b.transform(x_test2.reshape(-1,x_test2.shape[3])) #reshaping to 2d for standard scaling

x_train2 = scaler_b.fit_transform(x_train2.reshape(-1,1)) #reshaping to 2d for standard scaling
x_test2 = scaler_b.transform(x_test2.reshape(-1,1)) #reshaping to 2d for standard scaling

scaler_l = StandardScaler()
y_train = scaler_l.fit_transform(y_train.reshape(-1,1)) #reshaping to 2d for standard scaling
y_test = scaler_l.transform(y_test.reshape(-1,1)) #reshaping to 2d for standard scaling

# %% [code cell 12]
print('x_train.shape:',x_train.shape)
print('x_train2.shape:',x_train2.shape)
print('y_train.shape:',y_train.shape)

print('x_test.shape:',x_test.shape)
print('x_test2.shape:',x_test2.shape)
print('y_test.shape:',y_test.shape)

# %% [code cell 13]
#Reshaping data to 3D for modeling
lat = 448
lon = 304
timestep1 = 12
timestep2 = 24
features = 5

x_train = reshape_features(x_train, NUM_TRAIN, timestep1, lat, lon, features) # reshaping to 3d for model
x_test = reshape_features(x_test, NUM_TEST, timestep1, lat, lon, features) # reshaping to 3d for model

x_train2 = reshape_features(x_train2, NUM_TRAIN, timestep2, lat, lon, features) # reshaping to 3d for model
x_test2 = reshape_features(x_test2, NUM_TEST, timestep2, lat, lon, features) # reshaping to 3d for model

y_train = reshape_outcome(y_train, NUM_TRAIN, lat, lon) # reshaping to 3d for model
y_test = reshape_outcome(y_test, NUM_TEST, lat, lon) # reshaping to 3d for model

# %% [code cell 14]
print('x_train.shape:',x_train.shape)
print('x_train2.shape:',x_train2.shape)
print('y_train.shape:',y_train.shape)

print('x_test.shape:',x_test.shape)
print('x_test2.shape:',x_test2.shape)
print('y_test.shape:',y_test.shape)

# %% [code cell 15]
def custom_mse(y_true, y_pred):
	y_pred_masked = tf.math.multiply(y_pred, y_land_mask)
	y_true_masked = tf.math.multiply(y_true, y_land_mask)
	squared_resids = tf.square(y_true_masked - y_pred_masked)
	loss = tf.reduce_mean(squared_resids)
	return loss

# %% [code cell 16]
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, ConvLSTM2D, BatchNormalization, UpSampling2D,MaxPooling2D, concatenate, Flatten, Reshape
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model, Model

# %% [code cell 17]
input1_shape = (12, 448, 304, 5)
input2_shape = (24, 448, 304, 5)

learning_rate=1e-4
filter_size=3
use_temp_scaling=False
n_output_classes=1
loss = "mse"
metrics = RootMeanSquaredError()

# %% [code cell 18]
    input1 = Input(shape=input1_shape)
    input2 = Input(shape=input2_shape)

    convlstm1 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, data_format="channels_last")(input1)
    
    conv1 = Conv2D(np.int(16), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(convlstm1)
    conv1 = Conv2D(np.int(16), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    bn1 = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    convlstm2 = ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, data_format="channels_last")(input2)
    conv2 = Conv2D(np.int(32), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(convlstm2)
    conv2 = Conv2D(np.int(32), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    bn2 = BatchNormalization(axis=-1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(np.int(64), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(np.int(64), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    bn3 = BatchNormalization(axis=-1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    up8 = Conv2D(np.int(32), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(pool3))
    merge8 = concatenate([bn3,up8], axis=3)
    conv8 = Conv2D(np.int(32), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(np.int(32), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    bn8 = BatchNormalization(axis=-1)(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn8)

    up9 = Conv2D(np.int(16), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2), interpolation='nearest')(bn8))
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = Conv2D(np.int(16), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(np.int(16), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(np.int(16), filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    output = Conv2D(n_output_classes, 1, activation='linear')(conv9)
        
    model = Model(inputs = [input1, input2], outputs = output)
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss, metrics = metrics)

# %% [code cell 19]
print(model.summary())

# %% [code cell 20]
# define early stopping callback
early_stopping = EarlyStopping(patience=100, restore_best_weights=True)

# fit model
history = model.fit(x=[x_train, x_train2], y=y_train,epochs=50,batch_size=2,validation_split=.2,verbose = 2)

# %% [code cell 21]
y_pred = model.predict([x_test,x_test2])

# %% [code cell 22]
# invert scaling for forecasted values 
inv_y_pred = scaler_l.inverse_transform(y_pred.reshape(-1,1))

# invert scaling for actual values
inv_y_test = scaler_l.inverse_transform(y_test.reshape(-1,1))

# %% [code cell 23]
inv_y_pred = inv_y_pred.reshape(len(y_pred),448,304)
print(inv_y_pred.shape)
inv_y_test = inv_y_test.reshape(len(y_test),448,304)
print(inv_y_pred.shape)

# %% [code cell 24]
# reshape y_land_mask
land_mask = np.load("data/y_land_mask_actual.npy",allow_pickle=True)
y_land_mask = land_mask.reshape(1, 448, 304)

# %% [code cell 25]
y_pred_masked = np.multiply(inv_y_pred, y_land_mask)
y_true_masked = np.multiply(inv_y_test, y_land_mask)

# %% [code cell 26]
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

rmse = sqrt(mean_squared_error(inv_y_test.flatten(), inv_y_pred.flatten()))
print('Test RMSE: %.3f' % rmse)

r_sq = r2_score(inv_y_test.flatten(), inv_y_pred.flatten())
print('Test R_Square: %.3f' % r_sq)

# %% [code cell 27]
#Post-Process RMSE
post_y = np.clip(y_pred_masked, a_min = 0, a_max = 100)
rmse1 = sqrt(mean_squared_error(y_true_masked.flatten(), post_y.flatten()))
print('Post-Processed RMSE: %.3f' % rmse1)

r_sq = r2_score(y_true_masked.flatten(), post_y.flatten())
print('Post-Processed R_Square: %.3f' % r_sq)

mae = mean_absolute_error(y_true_masked.flatten(), post_y.flatten())
print('Post-Processed MAE: %.3f' % mae)

# %% [code cell 28]
with open("data/pred_ice_mt_icnet_lag6.npy", "wb") as f:
  np.save(f, post_y)

# %% [code cell 29]
from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize=(8,6))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='valid')
pyplot.legend()
pyplot.show()


# %% [entry]
if __name__ == "__main__":
    # If the notebook defined a `main()` we call it; otherwise, nothing happens.
    try:
        main  # type: ignore
    except NameError:
        pass
    else:
        main()
