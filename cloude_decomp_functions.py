# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:01:13 2022

@author: forschner
"""
import numpy as np
import rgb950_functions as rgb
import matplotlib.pyplot as plt

def M2C(MM):
    # sigma0 = [[1,0],[0,1]]
    # sigma1 = [[1,0],[0,-1]]
    # sigma2 = [[0,1],[1,0]]
    # sigma3 = [[0,-1j],[1j,0]]
    # sigma = [sigma0,sigma1,sigma2,sigma3]
    # U = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0, 1j, -1j, 0]])
    # PI = np.zeros((4,4,4,4)).astype(np.complex64)
    # for n in range(4):
    #     for mm in range(4):
    #         PI[n,mm,:,:] = 0.5*U@(np.kron(sigma[n],np.conj(sigma[mm])))@np.matrix(U).H;
    # PI=np.reshape(PI,(16,16));
    PI = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.-1.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.-1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+1.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j]], dtype=np.complex64)
    COH = 0.25*np.reshape(MM,[1,16])@PI;
    COH = np.reshape(COH,(4,4));
    return COH

def M2Cbig(MM):
    PI = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.-1.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.-1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+1.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j]], dtype=np.complex64)
    
    COH = 0.25*np.reshape(MM,[16,360_000]).T@PI.T
    COH = np.reshape(COH,(360_000,4,4))
    return COH

def C2M(COH):
    # sigma0 = [[1,0],[0,1]]
    # sigma1 = [[1,0],[0,-1]]
    # sigma2 = [[0,1],[1,0]]
    # sigma3 = [[0,-1j],[1j,0]]
    # sigma = [sigma0,sigma1,sigma2,sigma3]
    # U = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0, 1j, -1j, 0]])
    # PI = np.zeros((4,4,4,4)).astype(np.complex64)
    # for n in range(4):
    #     for mm in range(4):
    #         PI[n,mm,:,:] = 0.5*U@(np.kron(sigma[n],np.conj(sigma[mm])))@np.matrix(U).H;
    # PI=np.reshape(PI,(16,16));  
    PI = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.-1.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.-1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+1.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j]], dtype=np.complex64)
    MM = np.reshape(COH,[1,16])@np.conjugate(PI.T)
    MM = np.reshape(MM,[4,4])
    return np.real(MM)

def C2Mbig(COH):
    PI = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.-1.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.-1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+1.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j]], dtype=np.complex64)
    
    MM = np.reshape(COH,[360_000,16])@np.conjugate(PI.T)
    MM = np.reshape(MM,[360_000,4,4])
    return np.real(MM)

def cloudeDecomp(MM):
    [xi,vec] = np.linalg.eig(M2C(MM));
    idx = xi.argsort()[::-1]
    xi= xi[idx]
    vec = vec[:,idx]
    mmBasis = np.zeros([4,4,4])
    for n in range(4):    
        mmBasis[n,:,:] = C2M(np.outer(vec[:,n], np.conjugate(vec[:,n])))
    return np.real(xi),mmBasis

def cloudeDecompbig(MM):
    
    # auto-normalize
    MM = MM/MM[0,:,:]
    
    PI = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.-1.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.-1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+1.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j]], dtype=np.complex64)
    
    [xi,vec] = np.linalg.eig(M2Cbig(MM))

    
    idx = xi.argsort()[:,::-1]
    xi= np.take_along_axis(xi, idx, axis=1)
    xi = np.transpose(xi, (1, 0))
    
    
    idx = np.asarray([idx, idx, idx, idx])
    idx = np.transpose(idx, (1, 0, 2))
    vec = np.take_along_axis(vec, idx, axis=-1)
    
    
    mmBasis = np.zeros([360_000,4,4,4], dtype=np.complex64)

    cohBasis  = np.asarray(list(map(ooter, vec)))
    cohBasis = np.reshape(cohBasis, [360_000, 4, 16])

    mmBasis = cohBasis @ np.conjugate(PI.T)
    mmBasis = np.reshape(mmBasis,[360_000,4,4,4])
    
    # Dimensions are as follows: Eigenvalue, MMx, MMy, Pixel#
    mmBasis = np.transpose(mmBasis, (1, 3, 2, 0))
    
    return np.real(xi.reshape((4, 600, 600))),np.real(mmBasis)

def cloudeDominant(MM):
    [xi,vec] = np.linalg.eig(M2C(MM));
    idx = xi.argsort()[::-1]
    xi= xi[idx]
    vec = vec[:,idx]
    return C2M(np.outer(vec[:,0], np.conjugate(vec[:,0])))


def coherencyEigenvalues(MM):
    if np.shape(MM)==(16,):
        MM = MM.reshape([4,4])
    MM = np.divide(MM,MM[0,0])
    MM[np.isnan(MM)]=0
    MM[np.isinf(MM)]=0
    [xi,vec] = np.linalg.eig(M2C(MM));
    idx = xi.argsort()[::-1]
    xi= xi[idx]
    return xi


def ooter(vec):
    
    cohBasis = np.zeros([4,4,4], dtype=np.complex64)
    for n in range(4):    
        cohBasis[n,:,:] = np.outer(vec[:,n], np.conjugate(vec[:,n]))
        
    return cohBasis
    
if __name__ == '__main__':
    
    mm = rgb.readMMbin('./data/mo_eyes_close_3.bin')
    # mm = mm/mm[0,:,:]
    
    
    PI = np.array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.-1.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.-1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,
             0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+1.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+1.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j, -1.+0.j],
           [ 0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.-1.j,
             0.+0.j,  0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,
             0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+1.j,  0.+0.j,  0.+0.j,  0.-1.j,  0.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
             1.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
             0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
             0.+0.j,  1.+0.j]], dtype=np.complex64)
    
    # [xi1,vec1] = np.linalg.eig(coh1)
    
    # taking real components, assuming mueller matrix is physical
    # xi1 = xi1.real
    # idx1 = xi1.argsort()[:,::-1]
    # xi2= np.take_along_axis(xi1, idx1, axis=1)
    
    
    
    # idx2 = np.asarray([idx1, idx1, idx1, idx1])
    # idx2 = np.transpose(idx2, (1, 0, 2))
    # vec2 = np.take_along_axis(vec1, idx2, axis=-1)
    
    
    # cohBasis  = np.asarray(list(map(ooter, vec2)))
    # cohBasis = np.reshape(cohBasis, [360_000, 4, 16])
    
    # mmBasis = cohBasis @ np.conjugate(PI.T)
    # mmBasis = np.reshape(mmBasis,[360_000,4,4,4])
    
    xi1, mmB1 = cloudeDecompbig(mm)
    
    
    mmEig0 = mmB1[0,:,:,:].reshape(4, 4, 600, 600)
    
    plt.figure()
    plt.imshow(xi1[0,:,:])
    
    rgb.MMImagePlot(mmEig0, -1, 1)
    
    
    # for n in range(4):    
    #     mmBasis[:,n,:,:] = C2M(np.outer(vec2[:,:,n], np.conjugate(vec2[:,:,n])))
    
    # Coh = M2Cbig(mm.reshape(16, 360_000))
    # for ii in range(600):
    #     print(ii)
    #     for jj in range(600):
            
    #         xi, mmB = cloudeDecomp(mm[:,ii,jj])
    #         xig[:,ii,jj] = xi
    
    
