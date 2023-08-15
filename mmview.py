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
import os

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
# wv_947 = [-4.15989, 132.725, -0.329298, -13.3397, 117.562]
wv_451 = [13.41, 143.13, -0.53, -17.02, 130.01]
wv_524 = [13.41, 120.00, -0.53, -17.02, 124.55]
wv_662 = [13.41, 94.780, -0.53, -17.02, 127.12]

normal = np.ones([16,600,600])
normal[0,:,:] = 0
data_loaded = {'Retardance Vector' : 0, 'Cloude Decomposition' : 0}
new_mm = 0
def callb_click(event):
    print('click at {} {}'.format(event.x, event.y))
    
def callb_ax(event):
    for i in range(4):
        if event.inaxes == axs[i]:
            rgb.MMImagePlot(mmBasis[i], -1, 1, 'Mueller Matrix for Eigenvalue {}'.format(i+1))

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().bind("<Button-1>", callb_click)
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def window_main():
    '''
    Creates a pysimplegui window for the program.
    '''
    
    select = [[sg.Input('rmmdLoad', key='rmmdLoad', enable_events=True, visible=False), sg.FileBrowse('RGB950 File', target='rmmdLoad', file_types=(('RMMD', '.rmmd'),('CMMI', '.cmmi'))), sg.Combo(('451nm', '524nm', '662nm', '947nm'), default_value='947nm', k='rmmd_wv', readonly=True)],
              [sg.Input('fileGot', key='fileGot', enable_events=True, visible=False), sg.FileBrowse('Select MMbinary', target='fileGot', file_types=(('Mueller Matrix Files','.bin'),), size=(20, 1))]]
    
    export = [[sg.Button('Export Data', k='export_button', expand_x=True)],
              [sg.Combo(('Retardance Vector', 'Cloude Decomposition', 'Cloude Dominant Process', ''), k='export_combo', readonly=True)],]
        
    loading = [[sg.Column(select), sg.Column(export)]]
    
    polar_diatt_frame = [[sg.Radio('Polarizance', k='Polarizance', group_id='polar_diatt', expand_x=True, default=True), sg.Radio('Diattenuation', k='Diattenuation', expand_x=True, group_id='polar_diatt')],
                         [sg.Button('DoLP / AoLP', expand_x=True, disabled=True), sg.Button('Magnitude', expand_x=True, disabled=True)]]
    
    mm_function_buttons = [
        [sg.Button('Mueller Matrix Plot', expand_x=True, disabled=True), sg.Text('Threshold:'), sg.Input('', size=(8,1), k='MM_thresh')], 
        [sg.Frame('MM Vectors', polar_diatt_frame, expand_x=True)],
        [sg.Button('Lu-Chipman Retardance', expand_x=True, disabled=True)], 
        [sg.Button('Cloude Decomposition', expand_x=True, disabled=True)],
        # [sg.Input('ClDc_save', key='ClDc_save', enable_events=True, visible=False), sg.FileSaveAs('Save Cloude Decomp', target = 'ClDc_save', file_types=(('Binary Files', '.bin'),), visible=False)], # Obsolete
        [sg.ProgressBar(360, orientation='horizontal', visible=False, k='prog', expand_x=True)], 
        [sg.Button('Linear Retardance', expand_x=True, visible=False), sg.Button('Retardance Magnitude', expand_x=True, visible=False)],]
    
    # total layout for window
    lay = [[sg.Frame('Data Handling', loading, expand_x=True)],
           [sg.Frame('Processing', mm_function_buttons, expand_x=True)],
           [],]
              
    return sg.Window('RGB950 Post Processing', layout=lay, resizable=True, finalize=True, keep_on_top=True)


def window_rmmd():  # Obsolete
    lay = [[sg.Input('rmmdLoad', key='rmmdLoad', enable_events=True, visible=False), sg.FileBrowse('Load RMMD File', target='rmmdLoad', file_types=(('RMMD', '.rmmd'),))],
           [sg.Button('RMMD Video', expand_x=True, disabled=True)],
           [sg.Button('Convert to MM Binary', expand_x=True, disabled=True)],]
    
    return sg.Window('RMMD Files', layout=lay, resizable=True, finalize=True, keep_on_top=True)

def window_fig():
    lay = [[sg.Canvas(k='-CANVAS-')]]
    
    return sg.Window('Graph', lay, finalize=True, element_justification='center')

win = window_main()

# Creating data and video directory on computer
if not os.path.exists('./data/'):
    os.makedirs('./data/')
    
if not os.path.exists('./rmmd_videos'):
    os.makedirs('./rmmd_videos')


# Event handling loop
while True:
    
    # read events from all open windows
    window, event, values = sg.read_all_windows(timeout_key = '__TIMEOUT__')
    
    # begin with exit protocol before other events
    if event == sg.WIN_CLOSED or event == 'Exit':
        if window == win: 
            window.close()
            break # ends program
        else:
            window.close()
    
    elif event == 'fileGot': # MMbinary file loading
        
        mm = rgb.readMMbin(values['fileGot'])
        mmName = values['fileGot'].split('/')[-1].strip('.bin')
        is_cbox = 0
        if mmName[:4] == 'cbox':
            is_cbox = 1
            
        new_mm = 1
        # print('Mueller Matrix Loaded')
        window['Mueller Matrix Plot'].update(disabled=False)
        window['Magnitude'].update(disabled=False)
        window['DoLP / AoLP'].update(disabled=False)
        window['Lu-Chipman Retardance'].update(disabled=False)
        window['Cloude Decomposition'].update(disabled=False)
        
    
        
    elif event == 'Mueller Matrix Plot': # Plotting Mueller matrix with thresholding
        try:
            thresh = float(values['MM_thresh'])
            rgb.MMImagePlot(mm, -thresh, thresh, is_cbox=is_cbox)
        except:
            rgb.MMImagePlot(mm, -1, 1, is_cbox=is_cbox)
    
    
    
    elif event == 'Cloude Decomposition':
        if new_mm == 1: # Recalculate if there is new data
            window['Mueller Matrix Plot'].update(disabled=True)
            window['Magnitude'].update(disabled=True)
            window['DoLP / AoLP'].update(disabled=True)
            window['Lu-Chipman Retardance'].update(disabled=True)
            window['Cloude Decomposition'].update(disabled=True)
            
            eigBasis, mmBasis = cdf.cloudeDecompbig(mm)
            data_loaded['Cloude Decomposition'] = 1
            
            new_mm = 0
                
            window['Mueller Matrix Plot'].update(disabled=False)
            window['Magnitude'].update(disabled=False)
            window['DoLP / AoLP'].update(disabled=False)
            window['Lu-Chipman Retardance'].update(disabled=False)
            window['Cloude Decomposition'].update(disabled=False)
            
        # Eigenvalues figure
        eigFig, axs = plt.subplots(2, 2)
        axs = axs.reshape((4))
        for i in range(4):
            im = axs[i].imshow(eigBasis[i,:,:], interpolation='none', norm = mpl.colors.LogNorm(vmin = 0.001, vmax = 1.0))
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            
            axs[i].set_title('Eigenvalue: {}'.format(i))
        
        eigFig.subplots_adjust(right=0.8)
        cbar_ax = eigFig.add_axes([0.85, 0.15, 0.05, 0.7])
        eigFig.colorbar(im, cax=cbar_ax)
        # eigFig.tight_layout()
        eigFig.canvas.mpl_connect('button_press_event', callb_ax)
        
        # For future inline plot implementaton
        # winfig = window_fig()
        # draw_figure(winfig["-CANVAS-"].TKCanvas, eigFig)
        
    
    elif event == 'Magnitude':
        rgb.plot_mag(mm, diatt=values['Diattenuation'])
        
    
    elif event == 'DoLP / AoLP':
        rgb.plot_aolp(mm, diatt=values['Diattenuation'])
    
    elif event == 'Lu-Chipman Retardance':
        
        window['prog'].update(visible=True)
        window['Mueller Matrix Plot'].update(disabled=True)
        window['Magnitude'].update(disabled=True)
        window['DoLP / AoLP'].update(disabled=True)
        window['Lu-Chipman Retardance'].update(disabled=True)
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
        window['Magnitude'].update(disabled=False)
        window['DoLP / AoLP'].update(disabled=False)
        window['Lu-Chipman Retardance'].update(disabled=False)
        window['Cloude Decomposition'].update(disabled=False)
        window['Linear Retardance'].update(visible=True)
        window['Retardance Magnitude'].update(visible=True)
        
        data_loaded['Retardance Vector'] = 1
        
    elif event == 'Linear Retardance':
        rgb.plot_retardance_linear(ret_vec)
        
    elif event == 'Retardance Magnitude':
        rgb.plot_retardance_mag(ret_vec)
        
    
    elif event == 'rmmdLoad': # Loading RGB950 files
        
        filePath = values['rmmdLoad'].split('/')
        fileName = filePath[-1]
        
        
        if fileName[-4:] == 'rmmd':
            fileName = fileName.strip('.rmmd')
            
            rmmd = rgb.readRMMD(values['rmmdLoad'], )
            
            rmmdVidName = values['rmmdLoad'].split('/')[-1]
            ani = rgb.animRMMD(rmmd, fileName)
            if values['rmmd_wv'] == '947nm':
                rgb.makeRMMDbin(values['rmmdLoad'], './data/{}.bin'.format(fileName), wv=wv_947)
            elif values['rmmd_wv'] == '662nm':
               rgb.makeRMMDbin(values['rmmdLoad'], './data/{}.bin'.format(fileName), wv=wv_662)
            elif values['rmmd_wv'] == '524nm':
               rgb.makeRMMDbin(values['rmmdLoad'], './data/{}.bin'.format(fileName), wv=wv_524) 
            elif values['rmmd_wv'] == '451nm':
                rgb.makeRMMDbin(values['rmmdLoad'], './data/{}.bin'.format(fileName), wv=wv_451) 
                
                
        elif fileName[-4:] == 'cmmi':
            fileName = fileName.strip('.cmmi')
            cmmi = rgb.makeMMbin(values['rmmdLoad'], './data/{}.bin'.format(fileName))
        
        
    elif event == 'export_button':
        if values['export_combo'] == 'Retardance Vector':
            if data_loaded['Retardance Vector'] == 1:
                ret_vec.tofile('./data/{}_retardance.bin'.format(mmName))
        elif values['export_combo'] == 'Cloude Decomposition':
            if data_loaded['Cloude Decomposition']:
                eigBasis.tofile('./data/{}_eigenvalues.bin'.format(mmName))
                mmBasis.tofile('./data/{}_mmBasis.bin'.format(mmName))
                
        elif values['export_combo'] == 'Cloude Dominant Process':
            if data_loaded['Cloude Decomposition']:
                mmBasis[0].tofile('./data/{}_mmDominant.bin'.format(mmName))
        
    else:
        print(event)
        
