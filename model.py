import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tools import winLen, make_win, imp

# Load inputs
from inputs import ws, dens, winName, epochs

name = 's1'
s1 = np.expand_dims(imp(int(dens/100),name),axis=1)
serie = s1

'''
# Scale Data
scaler = MinMaxScaler(feature_range=(-1,1))
scaled_data = scaler.fit_transform(data)
filename = './results/scaler_' + str(nx) + 'x' + str(ny) + '.npz'
np.savez(filename, scalermin = scaler.data_min_, scalermax = scaler.data_max_)
'''
serie_len = serie.shape[0]
n_states = 1

xtrain = np.empty((0,ws,n_states), float)
ytrain = np.empty((0,n_states), float)
for i in range(winLen(),serie_len):
    winx = np.expand_dims(make_win(i,serie),axis=0)
    winy = np.expand_dims(serie[i,:],axis=0)
    xtrain = np.vstack((xtrain, winx))
    ytrain = np.vstack((ytrain, winy))


#Shuffling data
seed(1) # this line & next, what will they affect qqq
tf.random.set_seed(0)
perm = np.random.permutation(xtrain.shape[0])
xtrain = xtrain[perm,:,:]
ytrain = ytrain[perm,:]

#create the LSTM architecture
model = Sequential()
model.add(LSTM(80, input_shape=(ws, n_states), return_sequences=True, activation='tanh'))
model.add(LSTM(80, input_shape=(ws, n_states), activation='tanh'))
model.add(Dense(n_states))

#compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

#run the model
history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=100,
        validation_split=0.20, verbose=1)

#evaluate the model
scores = model.evaluate(xtrain, ytrain, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
filename = 'results/loss.png'
plt.savefig(filename, dpi = 200)
plt.show()

#Removing old models
model_name = './results/models/'+str(dens)+'_'+str(ws)+'_'+str(winName)+'.h5'
if os.path.isfile(model_name):
   os.remove(model_name)
#Save the model
model.save(model_name)
