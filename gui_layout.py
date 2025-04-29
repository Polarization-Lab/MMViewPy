import numpy as np
import PySimpleGUI as sg


def window_main():
    '''
    Creates a pysimplegui window for the program.
    '''
    # Threshold values for Min/Max inputs
    thresh_list = [round(i, 1) for i in np.arange(0, 1.1, 0.1)]
    
    
    load = [
        [
            sg.Text('Wavelength:'), 
            sg.Combo(
                ('451nm', '524nm', '662nm', '947nm'), 
                default_value='947nm', 
                key='rmmd_wv',
                size=(12,1),
                readonly=True, 
                expand_x=True
            ),
            sg.Checkbox(
                'Image Registration', 
                k='register_img', 
                size=(20,1),
                default=False, 
                expand_x=True
            )
        ],
        [
            sg.Input('rmmdLoad', key='rmmdLoad', enable_events=True, visible=False), 
            sg.FileBrowse(
                'RGB950 File', 
                target='rmmdLoad', 
                file_types=(('RMMD', '.rmmd'),('CMMI', '.cmmi')), 
                size=(27,1)
            ),
            sg.Input('fileGot', key='fileGot', enable_events=True, visible=False), 
            sg.FileBrowse(
                'Select MMbinary', 
                target='fileGot', 
                file_types=(('Mueller Matrix Files','.bin'),), 
                size=(27,1)
            )
        ]
    ]
    
    polar_diatt_frame = [
        [
            sg.Radio(
                'Polarizance', 
                key='Polarizance', 
                group_id='polar_diatt',
                expand_x=True, 
                default=True
            ), 
            sg.Radio(
                'Diattenuation', 
                key='Diattenuation', 
                group_id='polar_diatt',
                expand_x=True
            )
        ],
        [
            sg.Button('DoLP / AoLP', size=(20,1), expand_x=True, disabled=True), 
            sg.Button('Magnitude', size=(20,1), expand_x=True, disabled=True)
        ]
    ]
    
    polariscope_frame = [
        [
            sg.Radio(
                'Coordinates', 
                key='Coordinates', 
                group_id='polariscope',
                size=(8,1),
                expand_x=True, 
                default=True
            ), 
            sg.Radio(
                'Stokes', 
                key='Stokes', 
                group_id='polariscope', 
                size=(8,1),
                expand_x=True
            ),
            sg.Text('Min:'), 
            sg.Spin(
                thresh_list, 
                initial_value=0.0, 
                size=(4,1), 
                key='Min', 
                expand_x=True
            ), 
            sg.Text('Max:'), 
            sg.Spin(
                thresh_list, 
                initial_value=1.0, 
                size=(4,1), 
                key='Max', 
                expand_x=True
            )
        ],
        [
            sg.Text('PSG:'), 
            sg.Input('', expand_x=True, size=(20,1), key='PSG'), 
            sg.Text('PSA:'), 
            sg.Input('', expand_x=True, size=(20,1), key='PSA')
        ],
        [
            sg.Button('Polariscope View', size=(20,1), expand_x=True, disabled=True), 
            sg.Button('PSG Only', size=(20,1), expand_x=True, disabled=True)
        ]
    ]
    
    export = [
        [
            sg.Combo(
                ('Retardance Vector', 
                 'Cloude Decomposition', 
                 'Cloude Dominant Process'), 
                key='export_combo', 
                expand_x=True, 
                readonly=True, 
                size=(20,1)
            ),
            sg.Button('Export', key='export_button', expand_x=True)
        ]
    ]
            
    mm_funcs = [
        [
            sg.Button('Mueller Matrix Plot', size=(34,1), expand_x=True, disabled=True), 
            sg.Text('Threshold:'), 
            sg.Spin(thresh_list, initial_value=1.0, key='MM_thresh', expand_x=True)
        ],
        [sg.Frame('Polariscope', polariscope_frame, expand_x=True)],
        [sg.Frame('MM Vectors', polar_diatt_frame, expand_x=True)],
        [
            sg.Button('Cloude Decomposition', size=(20,1), expand_x=True, disabled=True),
            sg.Button('Lu-Chipman Retardance', size=(20,1), expand_x=True, disabled=True)], 
        [
            sg.ProgressBar(
                360, 
                orientation='horizontal', 
                key='prog', 
                visible=False, 
                expand_x=True
            )
        ], 
        [
            sg.Button('Linear Retardance', size=(20,1), expand_x=True, visible=False), 
            sg.Button('Retardance Magnitude', size=(20,1), expand_x=True, visible=False)
        ]
    ]
        
    # total layout for window
    title_font = ('Arial', 12)
    window_layout = [
        [sg.Frame('Data Handling', load, expand_x=True, font=title_font)],
        [sg.Frame('Processing', mm_funcs, expand_x=True, font=title_font)],
        [sg.Frame('Export Data', export, expand_x=True, font=title_font)]
    ]
              
    return sg.Window(
        'RGB950 Post Processing', 
        layout=window_layout, 
        resizable=True, 
        finalize=True
    )
