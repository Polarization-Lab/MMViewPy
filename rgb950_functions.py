# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:46:23 2022

@authors: qjare, forschner
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import ListedColormap
from rgb950_reconstruction import W_mat

plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = anim.FFMpegWriter(fps=10)

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


# wavelength params = [theta_A, delta_A, theta_LP, theta_G, delta_G]
wv_947 = [102.894, 68.2212, -1.17552, -104.486, 122.168]
# wv_947 = [-4.15989, 132.725, -0.329298, -13.3397, 117.562]
wv_451 = [13.41, 143.13, -0.53, -17.02, 130.01]
wv_524 = [13.41, 120.00, -0.53, -17.02, 124.55]
wv_662 = [13.41, 94.780, -0.53, -17.02, 127.12]

params = [wv_451, wv_524, wv_662, wv_947]

def readRMMD(inputfilename):
    raw = np.fromfile(inputfilename,np.int32).newbyteorder('>');
    [num,xdim,ydim]=raw[1:4]
    out=np.reshape(raw[5:5+num*xdim*ydim],[num,xdim,ydim])
    out = np.flip(out, axis=1)
    return out

def readCMMI(inputfilename):
    raw = np.fromfile(inputfilename,np.float32).newbyteorder('>');
    M = np.zeros([16,600,600])
    for i in range(16):
        M[i,:,:] = np.flipud(raw[5+i::16][0:(600*600)].reshape([600,600]).T)
    return M

def makeRMMDbin(inputfilepath, outputfilepath, wv = wv_947, psg_rot=False):
    '''

    Parameters
    ----------
    inputfilename : string
        Full path to input rmmd file.
    outputfilename : string
        Path for binary file output.
    wv : list, optional
        Five index list of wavelength parameters. See the list at top of rgb950_functions for examples. The default is wv_947.
    psg_rot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    rmmd = readRMMD(inputfilepath)
    n_len = len(rmmd[:,0,0])-1
    rmmd = np.float32(np.reshape(rmmd[:n_len, :, :], [n_len, 360_000]))
    W_plus = np.float32(np.linalg.pinv(W_mat(wv, n_len, psg_rot=psg_rot)))
    mm = np.matmul(W_plus, rmmd).reshape([16,600,600])
    mm.tofile(outputfilepath)

def makeMMbin(inputfilename,outputfilename):
    raw = np.fromfile(inputfilename,np.float32).newbyteorder('>');
    M = np.zeros([16,600,600],np.float32)
    for i in range(16):
        M[i,:,:] = np.flipud(raw[5+i::16][0:(600*600)].reshape([600,600]).T)
    M2=np.reshape(M,(16*600*600))
    M2.tofile(outputfilename)
    
def readMMbin(inputfilename):
    raw = np.fromfile(inputfilename,np.float32);
    M = np.reshape(raw,[16,600,600])
    return M

def MMImagePlot(MM,minval=-1,maxval=1, title='', is_cbox = 0, colmap=colmap):
    f, axarr = plt.subplots(nrows = 4,ncols = 4,figsize=(6, 5))
    f.suptitle(title, fontsize=20)
    
    MM = MM.reshape([4,4,600,600])
    
    if is_cbox == 1:
        MM = np.transpose(np.flipud(np.transpose(MM, (3, 2, 0, 1))), (2, 3, 0, 1))
    #normalization
    MM = MM/MM[0,0,:,:]
    
    for i in range(4):
        for j in range(4):
            im=axarr[i,j].imshow(MM[i,j,:,:],cmap=colmap,vmin = minval,vmax=maxval)
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            # im=axarr[i,j].imshow(MM[i,j,:,:], cmap=colmap)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    # plt.tight_layout()

def get_polarizance(MM):
    '''returns polarizance magnitude and orientation of input mueller matrix'''
    P = MM[[0,4,8],:,:]
    
    lin_mag = np.sqrt(P[1]**2 + P[2]**2)/P[0]
    lin_orient = 0.5 * np.arctan2(P[2], P[1])
    
    return lin_mag, lin_orient

def get_diattenuation(MM):
    P = MM[:4,:,:]
    
    lin_mag = np.sqrt(P[1]**2 + P[2]**2)/P[0]
    lin_orient = 0.5 * np.arctan2(P[2], P[1])
    
    return lin_mag, lin_orient

def updateAnim(frame, rmmd, im, ax):
    im.set_array(rmmd[frame])
    ax.set_title('{}'.format(frame))
    return im

def animRMMD(rmmd, outputfilename = 'recent_animation'):
    '''
    Animates Sequential images contained within an RMMD file. 
    
    [Input]
        rmmd : (string) path from program to rmmd file to open.
    
    [Output]
        Matplotlib Animated plot of sequential eye images.
    '''
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(outputfilename)
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(rmmd[0], animated=True)
    fig.colorbar(im)
    ani = anim.FuncAnimation(fig, updateAnim, 40, fargs=(rmmd, im, ax))
    ani.save('./rmmd_videos/{}.mp4'.format(outputfilename), writer = FFwriter)
    return ani

    


def RetardanceVector(MM):
    m00 = MM[0,0]
    M = MM/m00
    D = M[0,1:]
    Dmag = np.linalg.norm(D)
    x = 1-Dmag**2
    x = np.where(x<0, 0, x)
    mD = np.sqrt(x)*np.identity(3) + (1-np.sqrt(x))*np.outer(D/Dmag,D/Dmag)
    MD = np.vstack((np.concatenate(([1],D)),np.concatenate((D[:,np.newaxis],mD),axis=1)))
    Mprime = M@np.linalg.inv(MD)
    PDelta = Mprime[1:,0]
    [l1,l2,l3] = np.linalg.eigvals(Mprime[1:,1:]@Mprime[1:,1:].T)
    if np.linalg.det(Mprime[1:,1:]) > 0:
        mDelta = np.real((np.linalg.inv(Mprime[1:,1:]@Mprime[1:,1:].T + (np.sqrt(l1*l2) + np.sqrt(l2*l3) + np.sqrt(l3*l1))*np.eye(3))@((np.sqrt(l1)+np.sqrt(l2)+np.sqrt(l3))*Mprime[1:,1:]@Mprime[1:,1:].T + np.sqrt(l1*l2*l3)*np.eye(3))))
    else:
        mDelta = -np.real((np.linalg.inv(Mprime[1:,1:]@Mprime[1:,1:].T + (np.sqrt(l1*l2) + np.sqrt(l2*l3) + np.sqrt(l3*l1))*np.eye(3))@((np.sqrt(l1)+np.sqrt(l2)+np.sqrt(l3))*Mprime[1:,1:]@Mprime[1:,1:].T + np.sqrt(l1*l2*l3)*np.eye(3))))
       
    MDelta = np.vstack((np.concatenate(([1],np.zeros(3))),np.concatenate((PDelta[:,np.newaxis],mDelta),axis=1)))
    MR = np.linalg.inv(MDelta)@Mprime;
    MR = np.vstack((np.concatenate(([1],np.zeros(3))),np.concatenate((np.zeros(3)[:,np.newaxis],MR[1:,1:]),axis=1)))
    mR = MR[1:,1:];
    Tr = np.trace(MR)/2 -1
    if Tr < -1:
        Tr = -1
    elif Tr > 1:
        Tr = 1
    R = np.arccos(Tr)
    Rvec = R/(2*np.sin(R))*np.array([np.sum(np.array([[0,0,0],[0,0,1],[0,-1,0]]) * mR), np.sum(np.array([[0,0,-1],[0,0,0],[1,0,0]]) * mR), np.sum(np.array([[0,1,0],[-1,0,0],[0,0,0]]) * mR)])
    return Rvec

def plot_aolp(MM, cmap='hsv', diatt = 0):
    MM = MM.reshape(4,4, 600, 600)
    if diatt == 1:
        S0 = MM[0,0,:,:]
        S1 = MM[0,1,:,:] / S0
        S2 = MM[0,2,:,:] / S0
    
    else:
    
        S0 = MM[0,0,:,:]
        S1 = MM[1,0,:,:] / S0
        S2 = MM[2,0,:,:] / S0
    AoLP = 0.5 * np.arctan2(S2, S1)
    
    fig, axs = plt.subplots(ncols = 2, figsize=(6,2))
    
    if diatt == 1:
        fig.suptitle('Diattenuation Vector')
    else:
        fig.suptitle('Polarizance Vector')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    DoLP = np.sqrt(S1**2 + S2**2)
    DoLP[np.isnan(DoLP)] = 0
    
    AoLP_unwrap = np.unwrap(AoLP,discont=np.pi/2)
    AoLP = np.rad2deg(AoLP_unwrap)
    AoLP = np.mod(AoLP, 180)
    
    im1 = axs[0].imshow(DoLP, cmap='gray', interpolation = 'none', norm = matplotlib.colors.LogNorm(vmin = 0.001, vmax = 1.00, clip=True))
    im2 = axs[1].imshow(AoLP, cmap=cmap, vmin = 0, vmax = 180, interpolation='none')
    axs[0].set_title('DoLP')
    axs[1].set_title('AoLP')
    cb1 = fig.colorbar(im1)
    cb1.ax.set_yticks([0.01, 0.1, 1.0], ['1 %', '10 %', '100 %'], fontsize=12)
    cb2 = fig.colorbar(im2)
    cb2.ax.set_yticks([0, 45, 90, 135, 180], [r'$0\degree$', r'$45\degree$', '$90\degree$', r'$135\degree$', r'$180\degree$'], fontsize=12)
    # cb.ax.set_yticks([-90, -45, 0, 45, 90], [r'$-90\degree$', r'$-45\degree$', '$0\degree$', r'$45\degree$', r'$90\degree$'], fontsize=12)
   
def plot_mag(MM, cmap='viridis', diatt=0, axtitle='Magnitude'):
    MM = MM.reshape(16, 600, 600)
    if diatt == 1:
        mag, lin = get_diattenuation(MM)
    else:
        mag, lin = get_polarizance(MM)
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title(axtitle)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(mag, cmap=cmap, norm = matplotlib.colors.LogNorm(vmin = 0.001, vmax = 1.00, clip=True), interpolation='none')
    cb = fig.colorbar(im, )
    cb.ax.set_yticks([0.01, 0.1, 1.0], ['1 %', '10 %', '100 %'], fontsize=12)


def plot_retardance_linear(ret_vec):
    
    ret_vec = ret_vec.reshape([600, 600, 3])
    lin_ret = np.zeros([600,600])
    major_axis = np.zeros([600,600])
    
    for i in range(600):
        for j in range(600):
            delta_H = ret_vec[i,j,0]
            delta_45 = ret_vec[i,j,1]
            lin_ret[i,j] = np.sqrt(delta_H**2+delta_45**2)
            major_axis[i,j] = 0.5*np.arctan2(delta_45, delta_H)
    
    fig, ax = plt.subplots(ncols = 2, figsize=(6,3))
    fig.suptitle('Linear Retardance')
    im1 = ax[0].imshow(lin_ret, cmap='hsv', vmin = 0, vmax = np.pi, interpolation='none')
    im2 = ax[1].imshow(major_axis, cmap='hsv', vmin = -np.pi/2, vmax = np.pi/2, interpolation='none')
    ax[0].set_title('Magnitude')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_title('Major Axis')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cb1 = fig.colorbar(im1,shrink=0.8)
    cb1.ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3 \pi}{4}$', r'$\pi$'], fontsize=12)
    cb2 = fig.colorbar(im2,shrink=0.8)
    cb2.ax.set_yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'], fontsize=12)
    

def plot_retardance_mag(ret_vec):
    
    ret_mag = np.zeros([360_000])
    for jj in range(len(ret_mag)):
        ret_mag[jj] = np.linalg.norm(ret_vec[jj,:])
    ret_mag = ret_mag.reshape([600, 600])
    
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title('Retardance Magnitude')
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(ret_mag, cmap='turbo', vmin = 0, vmax = np.pi, interpolation='none')
    cb = fig.colorbar(im,)
    cb.ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3 \pi}{4}$', r'$\pi$'], fontsize=12)

    
def coordinates2stokes(lat_long):
    # Convert to radians
    latitude = np.radians(lat_long[0])
    longitude = np.radians(lat_long[1])
    
    # Convert longitude/latitude to stokes
    s0 = 1.0
    s1 = np.cos(latitude) * np.cos(longitude)
    s2 = np.cos(latitude) * np.sin(longitude)
    s3 = np.sin(latitude)
    stokes = np.array([s0, s1, s2, s3])
    
    return stokes


def run_polariscope(MM, PSA, PSG, vmin, vmax, PSG_str, PSA_str, use_coords):
    MM = MM.reshape([4,4,600,600])
    MM /= MM[0,0,:,:]
    S0_prime = np.zeros([600,600])  
    PSG = PSG.reshape((4, 1))
    
    for i in range(600):
        for j in range(600):
            S0_prime[i,j] = np.dot(0.5*PSA, MM[:,:,i,j]@PSG)
    
    # Plot S0_prime
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(S0_prime, cmap='gray', vmin=vmin, vmax=vmax, interpolation='none')
    cb = fig.colorbar(im)
    fig.suptitle(r'Polariscope View $S_0$')
    if use_coords == 1:
        ax.set_title(f'PSG lat/long: ({PSG_str})   PSA lat/long: ({PSA_str})', fontsize=10)
    else:
        ax.set_title(f'PSG Stokes: ({PSG_str})   PSA Stokes: ({PSA_str})', fontsize=10)
    
    # Poincare Sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    xgrid = np.outer(np.cos(u), np.sin(v))
    ygrid = np.outer(np.sin(u), np.sin(v))
    zgrid = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xgrid, ygrid, zgrid, color='gray', alpha=0.5, linewidth=0.5)
    ax.scatter(PSG[1], PSG[2], PSG[3], color='b', s=100, label='PSG')
    ax.scatter(PSA[1], PSA[2], PSA[3], color='r', s=100, label='PSA')
    
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel(r'$S_1$')
    ax.set_ylabel(r'$S_2$')
    ax.set_zlabel(r'$S_3$')
    ax.set_title('Poincare Sphere')
    ax.legend()
    ax.grid(False)
    plt.show()
    
    
def run_sim_psg(MM, PSG, colmap):
    MM = MM.reshape([4,4,600,600])
    MM = MM/MM[0,0,:,:]
    product = np.zeros([4,600,600])  
    PSG = np.squeeze(PSG)
    vmin = -1
    vmax = 1
    
    # Multiply MM by PSG vector
    for i in range(600):
        for j in range(600):
            product[:,i,j] = MM[:,:,i,j]@PSG
    
    # Plot results
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12,3))
    im0 = axs[0].imshow(product[0,:,:], cmap=colmap, vmin=vmin, vmax=vmax, interpolation='none')
    im1 = axs[1].imshow(product[1,:,:], cmap=colmap, vmin=vmin, vmax=vmax, interpolation='none')
    im2 = axs[2].imshow(product[2,:,:], cmap=colmap, vmin=vmin, vmax=vmax, interpolation='none')
    im3 = axs[3].imshow(product[3,:,:], cmap=colmap, vmin=vmin, vmax=vmax, interpolation='none')

    axs[0].set_title(r'$S_0$')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_title(r'$S_1$')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].set_title(r'$S_2$')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[3].set_title(r'$S_3$')
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im1, cax=cax)