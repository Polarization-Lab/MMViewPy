# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:04:35 2022

@author: isola7ion
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import ListedColormap
import rgb950_functions as rgb
import scipy.linalg as slin
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg


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


def window_main():
    '''
    Creates a pysimplegui window for the program.
    '''
    
    # start with interior columns
    # txts = [[sg.Text('X Freq Modifier:', expand_y=True)],
    #         [sg.Text('Y Freq Modifier:', expand_y=True)]]
    
    # slides = [[sg.Slider((1,9), default_value = 2, disable_number_display=True, expand_x=True, tick_interval=1, orientation='horizontal', enable_events=True, key='a')],
    #           [sg.Slider((1,9), default_value = 3, disable_number_display=True, expand_x=True, tick_interval=1, orientation='horizontal', enable_events=True, key='b')]]
    
    # cluster columns and visualization canvas in frame for ~ a e s t h e t i c s ~
    fra = [
            [sg.FileBrowse('Select File', enable_events=True, ), sg.Button('Exit', expand_x=True)]]
    
    # total layout for window
    lay = [[sg.Frame('RGB950', fra)]]
              
    return sg.Window('RGB950 Post Processing', layout=lay, resizable=True, finalize=True)

win = window_main()


while True:
    
    # read events from all open windows, which happens to just be variable 'win' for now
    window, event, values = sg.read_all_windows(timeout_key = '__TIMEOUT__')
    
    # begin with exit protocol before other events
    if event == sg.WIN_CLOSED or event == 'Exit':
        
        win.close()
        break
    
    
    else:
        print(event)
        
