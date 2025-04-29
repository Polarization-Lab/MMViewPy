# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:04:35 2022

@author: isola7ion
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import PySimpleGUI as sg
import os
import traceback

import rgb950_functions as rgb
import cloude_decomp_functions as cdf
from gui_layout import window_main


data_loaded = {'Retardance Vector' : 0, 'Cloude Decomposition' : 0}
new_mm = 0

# Creating data and video directory on computer
if not os.path.exists('./data/'):
    os.makedirs('./data/')
    
if not os.path.exists('./rmmd_videos'):
    os.makedirs('./rmmd_videos')

sg.theme('Dark Grey 10')
win = window_main()

# Event handling loop
try:
    while True:
        # read events from all open windows
        window, event, values = sg.read_all_windows(timeout_key = '__TIMEOUT__')
        
        # begin with exit protocol before other events
        if event in (sg.WINDOW_CLOSED, 'Exit'):
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
            window['Polariscope View'].update(disabled=False)
            window['PSG Only'].update(disabled=False)
            window['Magnitude'].update(disabled=False)
            window['DoLP / AoLP'].update(disabled=False)
            window['Lu-Chipman Retardance'].update(disabled=False)
            window['Cloude Decomposition'].update(disabled=False)
            
        elif event == 'Mueller Matrix Plot': # Plotting Mueller matrix with thresholding
            try:
                thresh = float(values['MM_thresh'])
                rgb.MMImagePlot(mm, -thresh, thresh, is_cbox=is_cbox)
            except:
                rgb.MMImagePlot(mm, is_cbox=is_cbox)
        
        elif event == 'Polariscope View':
            PSG_str = values['PSG']
            PSA_str = values['PSA']
            PSG = np.fromstring(values['PSG'], dtype=float, sep=',')
            PSA = np.fromstring(values['PSA'], dtype=float, sep=',')
            use_coords = values['Coordinates']
            if use_coords == 1:
                PSG = rgb.coordinates2stokes(PSG)
                PSA = rgb.coordinates2stokes(PSA)
            rgb.run_polariscope(mm, PSA, PSG, PSG_str, PSA_str, use_coords, float(values['Min']), float(values['Max']))
        
        
        elif event == 'PSG Only':
            PSG = np.fromstring(values['PSG'], dtype=float, sep=',')
            if values['Coordinates'] == 1:
                PSG = rgb.coordinates2stokes(PSG)
            rgb.run_sim_psg(mm, PSG)
        
        
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
                im = axs[i].imshow(eigBasis[i,:,:], interpolation='none', norm=mpl.colors.LogNorm(vmin=0.001, vmax=1.0))
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_title('Eigenvalue: {}'.format(i))
            
            eigFig.subplots_adjust(right=0.8)
            cbar_ax = eigFig.add_axes([0.85, 0.15, 0.05, 0.7])
            eigFig.colorbar(im, cax=cbar_ax)
            # eigFig.tight_layout()
            eigFig.canvas.mpl_connect('button_press_event', 
                                      lambda event: [
                                          rgb.MMImagePlot(mmBasis[i], -1, 1, 'Mueller Matrix for Eigenvalue {}'.format(i+1)) 
                                          for i in range(4) 
                                          if event.inaxes == axs[i]  # Check if the click was inside the i-th axis
                                          ]
                                      )
            
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
            window['Linear Retardance'].update(visible=True)
            window['Retardance Magnitude'].update(visible=True)
            window['Cloude Decomposition'].update(disabled=False)
        
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
                rmmd = rgb.readRMMD(values['rmmdLoad'], values['register_img'])
            
                rmmdVidName = values['rmmdLoad'].split('/')[-1]
                ani = rgb.animRMMD(rmmd, fileName)      

                rmmd = rgb.makeRMMDbin(
                    values['rmmdLoad'], 
                    './data/{}.bin'.format(fileName), 
                    values['rmmd_wv'], 
                    values['register_img']
                )
                
            elif fileName[-4:] == 'cmmi':
                fileName = fileName.strip('.cmmi')
                cmmi = rgb.makeMMbin(
                    values['rmmdLoad'], 
                    './data/{}.bin'.format(fileName), 
                )
            
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
            
except Exception:
    error_message = traceback.format_exc()
    sg.popup_error(f"An error occurred:\n{error_message}", title='ERROR', keep_on_top=True)
    window.close()

finally:
    window.close()