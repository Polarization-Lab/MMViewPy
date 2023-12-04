# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:48:58 2022

@author: forschner
"""

import numpy as np
import matplotlib.pyplot as plt


# Constants
# wv_947 = [102.894, 68.2212, -1.17552, -104.486, 122.168]
wv_947 = [-4.15989, 132.725, -0.329298, -13.3397, 117.562]
th_g = -104.486
th_a = 102.894
th_lp = -1.17552

d_g = 122.168
d_a = 68.2212



def LR(th,d):
    '''
    

    Parameters
    ----------
    d : FLOAT
        delta value in degrees
    th : FLOAT
        theta value in degrees

    Returns
    -------
    lr : NDARRAY [4x4]
        Linear retarder matrix

    '''
    
    # conversion to radians and precomputing values
    thrad = np.radians(th)
    drad = np.radians(d)
    cos2th = np.cos(2*thrad)
    sin2th = np.sin(2*thrad)
    cosdel = np.cos(drad)
    sindel = np.sin(drad)
    
    # generating matrix elements
    lr = np.zeros([4,4])
    lr[0,0] = 1.0
    lr[1,1] = cos2th**2 + cosdel*sin2th**2
    lr[2,1] = cos2th*sin2th*(1-cosdel)
    lr[3,1] = sindel*sin2th
    lr[1,2] = 1*lr[2,1]
    lr[2,2] = cosdel*cos2th**2+sin2th**2
    lr[3,2] = - cos2th*sindel
    lr[1,3] = -1*lr[3,1]
    lr[2,3] = -1*lr[3,2]
    lr[3,3] = cosdel
    
    return lr

def LP(th):
    '''
    

    Parameters
    ----------
    th : FLOAT
        theta value in degrees

    Returns
    -------
    lp : NDARRAY [4x4]
        Linear polarizer matrix

    '''
    thrad = np.radians(th)
    
    cos2th = np.cos(2*thrad)
    sin2th = np.sin(2*thrad)
    
    
    
    lp = np.zeros([4,4])
    lp[0,0] = 1.0
    lp[1,0] = cos2th
    lp[2,0] = sin2th
    lp[0,1] = cos2th
    lp[1,1] = cos2th**2
    lp[2,1] = cos2th*sin2th
    lp[0,2] = lp[2,0]
    lp[1,2] = lp[2,1]
    lp[2,2] = sin2th**2
    
    lp = 0.5 * lp
    
    return lp

def G_mat(n, th_g = -104.486, d_g = 122.168, step=9.0, g_bool = False):
    if g_bool == False:
        G = np.matmul(LR(n * step + th_g, d_g),LP(0.0))
    
    else:
        G = np.matmul(LR(n * step*4.91 + th_g, d_g),LP(0.0))
    
    return G



def A_mat(n, th_a = 102.894, d_a = 68.2212, th_lp = -1.17552, step=9.0, g_bool=False):
    if g_bool == False:
        A = np.matmul(LP(th_lp), LR(n*step*4.91 + th_a, d_a))
    else:
        A = np.matmul(LP(th_lp), LR(n*step + th_a, d_a))
    
    return A
    

def w_n(n, step, wv, g_bool = False):
    
    
    a = A_mat(n, th_a=wv[0], d_a=wv[1], th_lp=wv[2],step=step, g_bool=g_bool)[0,:]
    g = G_mat(n, th_g=wv[3], d_g=wv[4], step=step, g_bool=g_bool)[:,0]
    
    w = np.kron(a, g.T)
    
    return w

def W_mat(wv_params, n_len=40, psg_rot = False):
    W = np.ones([n_len,16])
    step = 360.0/n_len
    for n in range(n_len):
        W[n,:] = w_n(n, step, wv_params, psg_rot)
        
    return W

wv_662 = [13.41, 94.780, -0.53, -17.02, 127.12]


# W_40 = W_mat(wv_662, 40)
# W_20 = W_mat(wv_662, 20)

# lin_pol = LP(45).reshape([16,1])
# P_40 = W_40@lin_pol
# P_20 = W_20@lin_pol
# L_40 = np.linalg.pinv(W_40)
# L_20 = np.linalg.pinv(W_20)

# F_40 = L_40@P_40
# F_20 = L_20@P_20

