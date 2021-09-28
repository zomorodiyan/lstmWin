from keras.metrics import mean_absolute_error
import numpy as np
a = np.array([1.0,2.0,3.0])
b = np.array([3.0,4.0,5.8])
mae = mean_absolute_error(a,b).numpy()
for i in range(100):
    print('mae',mae)
