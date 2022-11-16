# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:06:59 2022

@author: forschner
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import ListedColormap
import rgb950_functions as rgb
import scipy.linalg as slin


#Blue to Red Color scale for S1 and S2
colmap = np.zeros((255,3));
# Red
colmap[126:183,0]= np.linspace(0,1,57);
colmap[183:255,0]= 1; 
# Green
colmap[0:96,1] = np.linspace(1,0,96);
colmap[158:255,1]= np.linspace(0,1,97); 
# Blue
colmap[0:71,2] = 1;
colmap[71:128,2]= np.linspace(1,0,57); 
colmap2 = colmap[128:,:]
colmap = ListedColormap(colmap)



'''
wavelength params = [theta_A, delta_A, theta_LP, theta_G, delta_G]
'''
wv_947 = [102.894, 68.2212, -1.17552, -104.486, 122.168]
wv_451 = [13.41, 143.13, -0.53, -17.02, 130.01]
wv_524 = [13.41, 120.00, -0.53, -17.02, 124.55]
wv_662 = [13.41, 94.780, -0.53, -17.02, 127.12]

normal = np.ones([16,600,600])
normal[0,:,:] = 0

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Execution below~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
m = 'mo_eyes_close_3'
# rmmd = rgb.readRMMD('./data/{}.rmmd'.format(m))
# rgb.makeRMMDbin('./data/{}.rmmd'.format(m), '{}.bin'.format(m), wv_947)

# mov = rgb.animRMMD(rmmd, '{}'.format(m))

mo = rgb.readMMbin('{}.bin'.format(m))


mo = mo.reshape([4,4,360000])
moRet = np.zeros([360000, 3])
for ii in np.arange(0, 360000):
    moRet[ii,:] = rgb.RetardanceVector(mo[:,:,ii])

moRet = moRet.reshape([600, 600, 3])
# rgb.MMImagePlot(mo, -1, 1)

fig = plt.figure()
im = plt.imshow(moRet[:,:,0], cmap=colmap, vmin = -np.pi, vmax = np.pi, interpolation='none')
cbar = fig.colorbar(im,)
# mo = mo/mo[0,0,:,:]

# rgb.lin_pol_ori(mo)



