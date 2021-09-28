import numpy as np
import os
from scipy.fftpack import dst, idst
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
from inputs import winName, ws, eqspace

def exp(dens,serie,name):
    folder = str(dens)
    if not os.path.exists('./results/'+folder):
        os.makedirs('./results/'+folder)
    filename = './results/'+folder+'/' + name +'.npz'
    np.savez(filename,serie=serie)

def imp(dens,name):
    folder = str(dens)
    filename = './results/'+folder+'/' + name +'.npz'
    data = np.load(filename)
    serie = data['serie']
    return serie

def percent(name,i,a,b):
    bar = ""
    for j in range(40):
        if(j<(i-a+1)/(b-a)*50): bar+='#'
        else: bar+='.'
    print(name,"{:.0f}".format((i-a+1)/(b-a)*100)+'%',bar,end='\r')
    if(i==b-1): print('')

def winLen():
    if winName == 'regular':
        return ws
    elif winName == 'eqspace':
        return ws*eqspace
    elif winName == 'addspace':
        length=0
        for i in range(ws): length += i+1
        return length
    else:
        raise ValueError("only name options:'regular','eqspace','addspace'")

def make_win(i,data):
    window = np.empty([ws,1])
    if winName == 'regular':
        window[:,:] = data[i-ws]
    elif winName == 'eqspace': # e.g. ws=3,eqspace=2 : x0x0x0y
        A = data[i-ws*eqspace:i-eqspace+1:eqspace]
        window[:,:] = A
    elif winName == 'addspace': # e.g. ws=3 : x00x0xy, ws=4 x000x00x0xy
        length=0
        for k in range(ws): length += k+1
        s = 0
        for j in range(1,ws+1):
            s += j
            window[-j,:] = data[i-s]
    else:
        raise ValueError("only name options:'regular','eqspace','addspace'")
    return window

def reconstruct_scaler():
# reconstruct the same scaler as the training time using saved min and max
    filename = './results/scaler.npz'
    data = np.load(filename)
    scalermin = data['scalermin']; scalermax = data['scalermax']
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit([scalermin,scalermax])
    return scaler
