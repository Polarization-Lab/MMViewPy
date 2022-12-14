# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:01:13 2022

@author: forschner
"""
import numpy as np
import cupy as cp
import rgb950_functions as rgb

def M2C(MM):
    sigma0 = [[1,0],[0,1]]
    sigma1 = [[1,0],[0,-1]]
    sigma2 = [[0,1],[1,0]]
    sigma3 = [[0,-1j],[1j,0]]
    sigma = [sigma0,sigma1,sigma2,sigma3]
    U = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0, 1j, -1j, 0]])
    PI = np.zeros((4,4,4,4)).astype(np.complex64)
    for n in range(4):
        for mm in range(4):
            PI[n,mm,:,:] = 0.5*U@(np.kron(sigma[n],np.conj(sigma[mm])))@np.matrix(U).H;
    PI=np.reshape(PI,(16,16));
    COH = 0.25*np.reshape(MM,[1,16])@PI;
    COH = np.reshape(COH,(4,4));
    return COH

def M2Cbig(MM):
    sigma0 = [[1,0],[0,1]]
    sigma1 = [[1,0],[0,-1]]
    sigma2 = [[0,1],[1,0]]
    sigma3 = [[0,-1j],[1j,0]]
    sigma = [sigma0,sigma1,sigma2,sigma3]
    U = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0, 1j, -1j, 0]])
    PI = np.zeros((4,4,4,4)).astype(np.complex64)
    for n in range(4):
        for mm in range(4):
            PI[n,mm,:,:] = 0.5*U@(np.kron(sigma[n],np.conj(sigma[mm])))@np.matrix(U).H;
    PI=np.reshape(PI,(16,16));
    COH = 0.25*PI@np.reshape(MM,[16,360_000]);
    COH = np.reshape(COH,(4,4, 360_000));
    return COH

def C2M(COH):
    sigma0 = [[1,0],[0,1]]
    sigma1 = [[1,0],[0,-1]]
    sigma2 = [[0,1],[1,0]]
    sigma3 = [[0,-1j],[1j,0]]
    sigma = [sigma0,sigma1,sigma2,sigma3]
    U = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0, 1j, -1j, 0]])
    PI = np.zeros((4,4,4,4)).astype(np.complex64)
    for n in range(4):
        for mm in range(4):
            PI[n,mm,:,:] = 0.5*U@(np.kron(sigma[n],np.conj(sigma[mm])))@np.matrix(U).H;
    PI=np.reshape(PI,(16,16));  
    MM = np.reshape(COH,[1,16])@np.conjugate(PI.T)
    MM = np.reshape(MM,[4,4])
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
    [xi,vec] = np.linalg.eig(M2Cbig(MM));
    idx = xi.argsort()[::-1]
    xi= xi[idx]
    vec = vec[:,idx]
    mmBasis = np.zeros([4,4,4])
    for n in range(4):    
        mmBasis[n,:,:] = C2M(np.outer(vec[:,n], np.conjugate(vec[:,n])))
    return np.real(xi),mmBasis

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

mm = rgb.readMMbin('./data/mo_eyes_close_3.bin')

xig = np.zeros((4, 360_000))
coh0 = M2C(mm[:,0,0])
coh1 = M2Cbig(mm[:,:,:])
# Coh = M2Cbig(mm.reshape(16, 360_000))
# for ii in range(600):
#     print(ii)
#     for jj in range(600):
        
#         xi, mmB = cloudeDecomp(mm[:,ii,jj])
#         xig[:,ii,jj] = xi


