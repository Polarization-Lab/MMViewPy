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
import cloude_decomp_functions as cdf
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

eigBasis = 0
mmBasis = 0

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def window_main():
    '''
    Creates a pysimplegui window for the program.
    '''
    
    loading = [
            [sg.Input('fileGot', key='fileGot', enable_events=True, visible=False), sg.FileBrowse('Select MM', target='fileGot', file_types=(('Mueller Matrix Files','.bin'),)), sg.Button('Raw Files', expand_x=True)]]
    
    mm_function_buttons = [
        [sg.Button('Mueller Matrix Plot', expand_x=True, disabled=True)], [sg.Button('Linear Polarizance Orientation', expand_x=True, disabled=True)],
        [sg.Button('Calc Retardance', expand_x=True, disabled=True)], 
        [sg.Button('Cloude Decomposition', expand_x=True, disabled=True)],
        [sg.ProgressBar(360, orientation='horizontal', visible=False, k='prog')], 
        [sg.Button('Lin Retardance Orientation', expand_x=True, visible=False), sg.Button('Retardance Magnitude', expand_x=True, visible=False)],]
    
    # total layout for window
    lay = [[sg.Frame('Load Data', loading, expand_x=True)],
           [sg.Frame('MM Plotting', mm_function_buttons, expand_x=True)],
           [],]
              
    return sg.Window('RGB950 Post Processing', layout=lay, resizable=True, finalize=True, keep_on_top=True)


def window_rmmd():
    lay = [[sg.Input('rmmdLoad', key='rmmdLoad', enable_events=True, visible=False), sg.FileBrowse('Load RMMD File', target='rmmdLoad', file_types=(('RMMD', '.rmmd'),))],
           [sg.Button('RMMD Video', expand_x=True, disabled=True)],
           [sg.Button('Convert to MM Binary', expand_x=True, disabled=True)],]
    
    return sg.Window('RMMD Files', layout=lay, resizable=True, finalize=True, keep_on_top=True)

def window_fig():
    lay = [[sg.Canvas(k='-CANVAS-')]]
    
    return sg.Window('Eigenvalues', lay, finalize=True, element_justification='center')

win = window_main()


while True:
    
    # read events from all open windows, which happens to just be variable 'win' for now
    window, event, values = sg.read_all_windows(timeout_key = '__TIMEOUT__')
    
    # begin with exit protocol before other events
    if event == sg.WIN_CLOSED or event == 'Exit':
        if window == win:
            
            window.close()
            break
        
        else:
            window.close()
    
    elif event == 'fileGot':
        
        mm = rgb.readMMbin(values['fileGot'])
        eigBasis = 0
        mmBasis = 0
        print('Mueller Matrix Loaded')
        window['Mueller Matrix Plot'].update(disabled=False)
        window['Linear Polarizance Orientation'].update(disabled=False)
        window['Calc Retardance'].update(disabled=False)
        window['Cloude Decomposition'].update(disabled=False)
        
    
        
    elif event == 'Mueller Matrix Plot':
        rgb.MMImagePlot(mm, -1, 1)
    
    elif event == 'Linear Polarizance Orientation':
        rgb.plot_lin_pol_ori(mm)
    
    elif event == 'Cloude Decomposition':
        if eigBasis == 0 & mmBasis == 0:
            window['Mueller Matrix Plot'].update(disabled=True)
            window['Linear Polarizance Orientation'].update(disabled=True)
            window['Calc Retardance'].update(disabled=True)
            window['Cloude Decomposition'].update(disabled=True)
            
            eigBasis, mmBasis = cdf.cloudeDecompbig(mm)
            
            window['Mueller Matrix Plot'].update(disabled=False)
            window['Linear Polarizance Orientation'].update(disabled=False)
            window['Calc Retardance'].update(disabled=False)
            window['Cloude Decomposition'].update(disabled=False)
        
        eigFig, axs = plt.subplots(2, 2)
        axs = axs.reshape((4))
        for i in range(4):
            axs[i].imshow(eigBasis[i,:,:])
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title('Eigenvalue: {}'.format(i+1))
            
        winfig = window_fig()
        
        draw_figure(winfig["-CANVAS-"].TKCanvas, eigFig)
        
    
    elif event == 'Calc Retardance':
        window['prog'].update(visible=True)
        window['Mueller Matrix Plot'].update(disabled=True)
        window['Linear Polarizance Orientation'].update(disabled=True)
        window['Calc Retardance'].update(disabled=True)
        window['Cloude Decomposition'].update(disabled=True)
        mm = mm.reshape([4,4,360_000])
        ret_vec = np.zeros([360_000, 3])
        for ii in np.arange(0, 360_000):
            ret_vec[ii,:] = rgb.RetardanceVector(mm[:,:,ii])
            if ii % 1000 == 0:
                window['prog'].update(current_count=ii//1000)
        mm = mm.reshape(16, 600, 600)
        window['prog'].update(current_count=0, visible=False)
        window['Mueller Matrix Plot'].update(disabled=False)
        window['Linear Polarizance Orientation'].update(disabled=False)
        window['Calc Retardance'].update(disabled=False)
        window['Cloude Decomposition'].update(disabled=False)
        window['Lin Retardance Orientation'].update(visible=True)
        window['Retardance Magnitude'].update(visible=True)
        
    elif event == 'Lin Retardance Orientation':
        rgb.plot_retardance_linear(ret_vec)
        
    elif event == 'Retardance Magnitude':
        rgb.plot_retardance_mag(ret_vec)
        
    elif event == 'Raw Files':
        rmmd_win = window_rmmd()
    
    elif event == 'rmmdLoad':
        rmmdName = values['rmmdLoad'].split('/')[-1].strip('.rmmd')
        rmmd = rgb.readRMMD(values['rmmdLoad'])
        
        window['RMMD Video'].update(disabled=False)
        window['Convert to MM Binary'].update(disabled=False)
        
    elif event == 'RMMD Video':
        rmmdVidName = values['rmmdLoad'].split('/')[-1]
        ani = rgb.animRMMD(rmmd, rmmdName)
    
    else:
        print(event)
        
