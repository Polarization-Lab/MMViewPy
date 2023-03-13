# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:50:32 2023

@author: forschner
"""

import numpy as np
import matplotlib.pyplot as plt
import rgb950_functions as rgb
import matplotlib.animation as anim
from matplotlib.colors import ListedColormap

plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = anim.FFMpegWriter(fps=10)

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

sphere_colors = ['green', 'red', 'white']
wavelengths = ['662', '524', '451']
angles = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]



def updateAnim(frame, path, im, ax, mask):
    angles = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]
    MM = rgb.readMMbin('{}\\{}.bin'.format(path, angles[frame]))
    MM = MM * mask
    MM = MM / MM[0,:,:]
    MM = MM.reshape(4,4,600,600)
    ax[0,0].set_title(angles[frame])
    for i in range(4):
        for j in range(4):
            im[i+4*j].set_array(MM[i,j,:,:])
            
    
    

def sphere_anim(path, outputfilename='sphere'):
    
    sphere_mask = np.zeros([16,600,600])
    sphere_mask[0,:,:] = 1
    for xi in range(600):
        for yi in range (600):
            if np.sqrt((xi-265)**2 + (yi-293)**2) < 180:
                sphere_mask[1:,xi, yi] = 1
    
    MM = rgb.readMMbin('{}\\20.bin'.format(path))
    MM = MM * sphere_mask
    MM = MM / MM[0,:,:]
    MM = MM.reshape(4,4,600,600)
    f, axarr = plt.subplots(nrows = 4,ncols = 4,figsize=(6, 5))
    f.suptitle(outputfilename, fontsize=20)
    im = {}
    for i in range(4):
        for j in range(4):
            im[i+4*j]=axarr[i,j].imshow(MM[i,j,:,:],cmap=colmap,vmin = -1,vmax=1, animated=True)
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im[i+4*j], cax=cbar_ax)
    
    ani = anim.FuncAnimation(f, updateAnim, 15, fargs=(path, im, axarr, sphere_mask))
    ani.save('./rmmd_videos/{}.mp4'.format(outputfilename), writer = FFwriter)
    return ani




for color in sphere_colors: 
    for wv in wavelengths:
        path = 'P:\\Projects\\Oculus\\Data Sample Library\\RGB950 Polarimeter\\Measurements\\2023\\forschner_cbox\\Spheres\\{}\\{}'.format(color,wv)
        a = sphere_anim(path, outputfilename='{}_{}'.format(color, wv))