import numpy as np
import matplotlib.pyplot as plt
from tools import imp
from inputs import dens,bgn,fin

import sys
args = sys.argv
if len(args) > 1:
    def err(): raise ValueError("args must be digits")
    if args[1].isdigit(): dens = int(args[1])
    else: err()
    if len(args) > 2:
        if args[2].isdigit(): bgn = int(args[2])
        else: err()
    if len(args) > 3:
        if args[3].isdigit(): fin = int(args[3])
        else: err()
else:
    print('no args provided, values from inputs.py')
print('dens:',dens,'bgn:',bgn,'fin:',fin)

t = np.linspace(bgn,fin,dens)
s1 = imp(int(dens/100),'s1')
s1p = imp(int(dens/100),'s1p')

fig, ax = plt.subplots(1,1,figsize=(10,10))
plt.suptitle('s1 evolution', fontsize=25)
ax.grid()
ax.plot(t, s1,linewidth=3, color='black')
ax.plot(t, s1p,linewidth=2, color='red')
#ax.plot(t, s1p,linewidth=3,linestyle='dashed', color='red')
ax.set(xlabel='t')
ax.set(ylabel='magnitude')
fig.savefig('./results/s1.png')
plt.show()
fig.clear(True)
