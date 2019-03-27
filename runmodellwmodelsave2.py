from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, GaussianNoise, Dropout, BatchNormalization
from keras import initializers
from keras.optimizers import SGD, Nadam, RMSprop
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing #
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
import keras.backend as kb
import keras.backend as K
import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

#from netCDF4 import Dataset

def rae(y_true, y_pred):
  return tf.reduce_sum(tf.abs(y_pred-y_true)) / tf.reduce_sum(tf.abs(y_true))

def mae(y_true, y_pred):
  return tf.reduce_mean(tf.abs(y_pred-y_true))

def mape(y_true, y_pred):
  diff = tf.abs((y_true - y_pred) / (tf.abs(y_true)))
  return 100. * tf.reduce_mean(diff)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# reading the input data

X_train0 = np.load('lwXTrainin.npy')
print (X_train0.shape)

#X_train0 = np.delete(X_train0,np.s_[0:63],1)
#X_train0 = np.delete(X_train0,np.s_[373:403],1)

Y_train0 = np.load('lwYTrainout.npy')
#Y_train0 = Y_train00[:,:]
print (Y_train0.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_train0,Y_train0, test_size=0.1, random_state=42)

print (X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

np.save('X_train.npy',X_train)
np.save('Y_train.npy',Y_train)
np.save('X_test.npy',X_test)
np.save('Y_test.npy',Y_test)

XMEAN=np.max(X_train,axis=0)
XSTDD=np.min(X_train,axis=0)

#for j in range(1363):
for j in range(1426):
 if((XMEAN[j]-XSTDD[j])== 0):
  X_train[:,j] = 0.0
  X_test[:,j] = 0.0
 else:
  X_train[:,j]=(X_train[:,j]-XSTDD[j])/(XMEAN[j]-XSTDD[j])
  X_test[:,j]=(X_test[:,j]-XSTDD[j])/(XMEAN[j]-XSTDD[j])
 
print (XMEAN[189:219])
print (XSTDD[189:219])
#np.savetxt('xdiffsw7.dat',XMEAN-XSTDD)
#np.savetxt('xminsw7.dat',XSTDD)
#np.savetxt('xmaxsw7.dat',XMEAN)

YMEAN=np.max(Y_train,axis=0)
YSTDD=np.min(Y_train,axis=0)

for j in range(1154):
 if(YMEAN[j]-YSTDD[j]== 0):
  Y_train[:,j] = Y_train[:,j]
 else:
  Y_train[:,j]=(Y_train[:,j]-YSTDD[j])/(YMEAN[j]-YSTDD[j])

#np.savetxt('ydiffsw7.dat',YMEAN-YSTDD)
#np.savetxt('yminsw7.dat',YSTDD)
#np.savetxt('ymaxsw7.dat',YMEAN)

print (X_train.shape, X_test.shape)
print (Y_train.shape, Y_test.shape)

#fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = Sequential()
model.add(Dense(32, input_dim=1426, kernel_initializer='uniform', activation = 'sigmoid'))
model.add(Dense(32,                 kernel_initializer='uniform', activation = 'sigmoid'))
model.add(Dense(32,                 kernel_initializer='uniform', activation = 'sigmoid'))
model.add(Dense(1154,               kernel_initializer='uniform'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

#estimators = KerasRegressor(model, epochs=1, batch_size=4096)
history=model.fit(X_train,Y_train, epochs=400, batch_size=256,validation_split=0.1)

print (X_train.shape, Y_train.shape)
score  = model.predict(X_test)
print (score.shape)

for j in range(1154):
 score[:,j]=score[:,j]*(YMEAN[j]-YSTDD[j])+YSTDD[j]

#summary = model.summary()
W_Input_Hidden0 = model.layers[0].get_weights()[0]; print (W_Input_Hidden0.shape)
biases0  = model.layers[0].get_weights()[1]; print (biases0.shape)
W_Input_Hidden1 = model.layers[1].get_weights()[0]; print (W_Input_Hidden1.shape)
biases1  = model.layers[1].get_weights()[1]; print (biases1.shape)
W_Input_Hidden2 = model.layers[2].get_weights()[0]; print (W_Input_Hidden2.shape)
biases2  = model.layers[2].get_weights()[1]; print (biases2.shape)
W_Input_Hidden3 = model.layers[3].get_weights()[0]; print (W_Input_Hidden3.shape)
biases3  = model.layers[3].get_weights()[1]; print (biases3.shape)

np.save('SWHidden001.npy',W_Input_Hidden0)
np.save('SWbiases001.npy',biases0)
np.save('SWHidden011.npy',W_Input_Hidden1)
np.save('SWbiases011.npy',biases1)
np.save('SWHidden021.npy',W_Input_Hidden2)
np.save('SWbiases021.npy',biases2)
np.save('SWHidden031.npy',W_Input_Hidden3)
np.save('SWbiases031.npy',biases3)
np.save('SWX_test1.npy', X_test)
np.save('SWY_test1.npy', Y_test)
np.save('SWScore1.npy', score)

print(history.history.keys())
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
np.save('train_loss.npy',train_loss)
np.save('val_loss.npy',val_loss)


