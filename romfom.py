import numpy as np
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow; silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS, podrec_svd, RK3t,\
     BoussRHS_t, export_data_test, export_data2, export_data3, show_percent,\
     reconstruct_scaler, make_win, winLen

# Load inputs
from inputs import lx, ly, nx, ny, Re, Ri, Pr, dt, nt, ns, freq, nr, eqspace, ws

#%%---------------------------define variables---------------------------------
abromfom = np.empty([ns+1,2*nr])
length = winLen()
az = np.empty([length,nr])
bz = np.empty([length,nr])
ablstm = np.empty([ns+1,2*nr])
xtest = np.empty([1,ws,2*nr])
dx = lx/nx; dy = ly/ny

#%%------------------------------load data-------------------------------------
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
wm = data['wm']; Phiw = data['Phiw'] # for ROM-FOM
tm = data['tm']; Phit = data['Phit']
sm = data['sm']; Phis = data['Phis']
aTrue = data['aTrue']; bTrue = data['bTrue'] # for ROM-ROM

scaler = reconstruct_scaler(nx,ny)
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')

# --------------------------------ROM-FOM--------------------------------------
w,s,t = initial(nx,ny) # init
for i in range(length): # first window_size steps

    show_percent('initFOM',i,0,length)
    export_data2(nx,ny,i,w,s,t,'romfom') # for romfom contour

    w_1d = w.reshape([-1,]); w_spread_1d = w_1d - wm #evaluate w_spread_1d
    az[i,:] = podproj_svd(w_spread_1d,Phiw) # get new a
    t_1d = t.reshape([-1,]); t_spread_1d = t_1d - tm #evaluate w_spread_1d
    bz[i,:] = podproj_svd(t_spread_1d,Phit) # get new b
    if(i != length-1): #run fom for freq. times; update w,s,t
        for j in range (freq):
            w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
abromfom[0:length,:] = scaler.transform(np.concatenate((az, bz), axis=1))


#======================================load closure============================
filenamec = './results/closure_' + str(nx) + 'x' + str(ny) + '.npz'
datac = np.load(filenamec)
wclosure= datac['wclosure']
sclosure= datac['sclosure']
tclosure= datac['tclosure']
#==============================================================================

for i in range(length, ns+1): # rest of steps after first window_size steps
    show_percent('ROMFOM',i,length,ns+1)

#Here instead of w,s,t I am gonna use w_1mode + wclosure
#==============================================================================
    '''
    mode_a = np.zeros([ns+1,nr])
    mode_a[0,:] = aTrue[0,:]
    mode_w_1d = podrec_svd(mode_a, Phiw)[:,i].reshape([-1,1])\
            + wm.reshape([-1,1]) # reconst. w
    mode_w = mode_w_1d.reshape([nx+1,-1]) #w1d to w2d
    w = mode_w + wclosure[i,:,:]

    mode_s_1d = podrec_svd(mode_a, Phis)[:,i].reshape([-1,1])\
            + sm.reshape([-1,1]) # reconst. s
    mode_s = mode_s_1d.reshape([nx+1,-1]) #w1d to w2d
    s = mode_s + sclosure[i,:,:]

    mode_b = np.zeros([ns+1,nr])
    mode_b[0,:] = bTrue[0,:]
    mode_t_1d = podrec_svd(mode_b, Phit)[:,i].reshape([-1,1])\
            + tm.reshape([-1,1]) # reconst. t
    mode_t = mode_t_1d.reshape([nx+1,-1]) #w1d to w2d
    t = mode_t + tclosure[i,:,:]
    '''
#==============================================================================


    t = RK3t(BoussRHS_t,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt*freq) #update t by fom_t
    abz = np.copy(make_win(i,abromfom))
    prediction = model.predict(np.expand_dims(abz, axis=0)) #run model on window
    a_new_iscl = scaler.inverse_transform(prediction)[:,0:nr] #extrct a form ab
    t_1d = t.reshape([-1,]) #t2d to t1d
    b_new_iscl = np.expand_dims(podproj_svd(t_1d - tm,Phit), axis=0)#proj b
    abromfom[i,:] = scaler.transform(np.concatenate((a_new_iscl,\
        b_new_iscl), axis=1)) #put ab_new at the end of abromfom

    w_1d = podrec_svd(a_new_iscl, Phiw) + wm.reshape([-1,1]) # reconst. w
    w = w_1d.reshape([nx+1,-1]) #w1d to w2d
    s_1d = podrec_svd(a_new_iscl, Phis) + sm.reshape([-1,1]) # reconst. s
    s = s_1d.reshape([nx+1,-1]) #s1d to s2d

    export_data2(nx,ny,i,w,s,t,'romfom') #for romfom contour

# --------------------------------ROM-ROM--------------------------------------
# Scale Data
ablstm[0:length,:] = scaler.transform(np.concatenate((az, bz), axis=1))

for i in range(length, ns+1):
    show_percent('ROM',i,length,ns+1)

    xtest = np.copy(np.expand_dims(make_win(i,ablstm), axis=0))
    ablstm[i,:] = np.copy(model.predict(xtest))

#%% export a, b for rom, fom, romfom
export_data3(nx,ny,ablstm,'rom')
export_data3(nx,ny,scaler.transform(np.concatenate((aTrue, bTrue), axis=1)),'fom')
export_data3(nx,ny,abromfom,'romfom')
