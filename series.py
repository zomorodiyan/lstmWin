import numpy as np
from tools import exp
from inputs import dens,bgn,fin

t = np.linspace(bgn,fin,dens)
s1 = np.empty([dens])
#s1 = 10*np.exp(-50*np.abs(t)) - 0.01/(np.power((t-0.5),2)+0.001)\
#        + 5*np.sin(5*t)
s1 = np.sin(t)

exp(int(dens/100),s1,'s1')

