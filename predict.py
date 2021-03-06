# use the first len timeSteps from the series itself
import numpy as np
from inputs import len,dens,winName,ws
from tools import exp, imp, winLen,percent, make_win, reconstruct_scaler
from tensorflow.keras.models import load_model
from silence_tensorflow import silence_tensorflow; silence_tensorflow()


name = 's1'
# load the first len time steps
serie = np.expand_dims(imp(int(dens/100),name),axis=1)
scaler = reconstruct_scaler()
serie = scaler.transform(serie)


model = load_model('./results/models/'+str(dens)+'_'+str(ws)+'_'+str(winName)+'.h5')
for i in range(winLen(), dens):
    percent('ROM',i,len,dens+1)
    win = np.copy(np.expand_dims(make_win(i,serie), axis=0))
    serie[i,:] = np.copy(model.predict(win))

#%% export a, b for rom, fom, romfom
serie = scaler.inverse_transform(serie)
exp(int(dens/100),serie,name+'p')
