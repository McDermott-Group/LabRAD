# HEMT read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import hemt_qubit_experiments

                          
comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Leiden Board DAC 3', 
                            'Leiden Board DAC 4',
                            'Leiden Board ADC 5'
                          ],
                'Leiden Board DAC 3': {
                                        'DAC A': 'Qubit Q',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                      },
                'Leiden Board DAC 4': {
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I',
                                      },
                'Leiden Board ADC 5': {
                                        'RunMode': 'demodulate', #'average'
                                        'FilterType': 'hann',
                                        'FilterWidth': 4000 * ns,
                                        'FilterLength': 4000 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -50 * MHz,
                                        'ADCDelay': 0 * ns,
                                        'Data': True
                                      },
                'Variables': {
                                'Init Time': {},
                                'Qubit SB Frequency': {'Value': 0 * MHz},
                                'Qubit Amplitude': {'Value': 0 * DACUnits},
                                'Qubit Time': {'Value': 0 * ns},
                                'Readout SB Frequency': {'Value': 0 * MHz},
                                'Readout Amplitude': {'Value': 0 * DACUnits},
                                'Readout Time': {'Value': 0 * ns},
                                'Qubit Drive to Readout Delay': {'Value': 0 * ns},
                                'ADC Wait Time': {'Value': 0 * ns},
                                # 'Stark Amplitude': {'Value': 0 * DACUnits},
                                # 'Stark Time': {'Value': 0 * ns},
                                'ADC Demod Frequency': {'Value': -50 * MHz}
                             }
                },
                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    'Variables': {  
                                    'Qubit Power': {'Setting': 'Power'}, 
                                    'Qubit Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {  
                                    'Readout Power': {'Setting': 'Power'}, 
                                    'Readout Frequency': {'Setting': 'Frequency'}
                                 }
                },
                # { # Lab RF Generator
                    # 'Interface': 'Lab Brick RF Generator',
                    # 'Serial Number': 10776,
                    # 'Variables': {  
                                    # 'Qubit Power': {'Setting': 'Power'}, 
                                    # 'Qubit Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7031,
                    'Variables': 'Readout Attenuation'
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': 'Qubit Attenuation'
                },
                { # SIM Voltage Source.
                    'Interface': 'SIM928 Voltage Source',
                    'Address': ('SIM900 - ' + comp_name + 
                                ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    'Variables': 'Qubit Flux Bias Voltage'
                },
                { # Leiden
                    'Interface': 'Leiden',
                    'Variables': {'Temperature': {'Setting': 'Mix Temperature'}}
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Reps', 'Runs'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'ADC5',
            'User': 'Guilhem Ribeill',
            'Base Path': 'Z:\mcdermott-group\Data\ADC Testing',
            'Experiment Name': 'BandpassFilter',
            'Comments': 'Driving with IQ modulation, test of Hittite HMC451 amplifier on DAC IQ mixer. NARDA + 2 MiniCircuits stacks in RT ampl. chain. 41-58 MHz band-pass filters on the ADC IQ-mixer.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 4000, # should not exceed ~5,000, use agrument "runs" in sweep parameters instead 

            'Init Time': 100 * us,
            
            # 'Stark Amplitude': 0 * DACUnits,
            # 'Stark Time': 10000 * ns,

            'Readout Frequency': 4.9135 * GHz, #4.919 * GHz,
            'Readout Power': 10 * dBm,
            'Readout Attenuation': 10 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 50 * MHz, # 41-58 MHz band-pass fileter on the ADC IQ-mixer.
            'Readout Amplitude': 0.5 * DACUnits,
            'Readout Time': 2000 * ns,
            
            'Qubit Drive to Readout Delay': 20 * ns,
            
            'Qubit Frequency': 20 * GHz,
            'Qubit Power': 13 * dBm,
            'Qubit Attenuation': 8 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0.9 * DACUnits,
            'Qubit Time': 0 * ns,
            # 'Qubit T2 Delay': 0 * ns,

            'Qubit Flux Bias Voltage': 0 * V,
                        
            'ADC Wait Time': 0 * ns, # time delay between the start of the readout pulse and the start of the demodulation
            # 'ADC Demod Frequency': -50 * MHz
           }

# with hemt_qubit_experiments.HEMTStarkShift() as run:
# with hemt_qubit_experiments.HEMTRamsey() as run:
with hemt_qubit_experiments.HEMTQubitReadout() as run:
#with hemt_qubit_experiments.ADCDemodTest() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    # run.sweep('ADC Demod Frequency', np.linspace(-100, 100, 101) * MHz,
        # print_data=['Amplitude'], plot_data=['Q', 'I', 'Amplitude'],
        # save=True, runs=1, max_data_dim=1)
        
    # run.single_shot_iqs(save=True, plot_data=True)
    #run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    #run.avg_osc(save=False, plot_data=['I', 'Q'], runs=2000)

    # run.sweep('Readout Attenuation', np.linspace(0, 63, 64) * dB,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=1, max_data_dim=1)
    
    # run.value('Readout Attenuation', -30.*dB)
    
    run.sweep('Readout Frequency', np.linspace(3, 5, 101) * GHz,
        print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        save=True, runs=1, max_data_dim=1)
    
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
      # [np.linspace(10, 40, 16) * dB, np.linspace(4.91, 4.92, 101) * GHz],
      # save=True, print_data='Amplitude', runs=1)
    
    # run.value('Qubit Amplitude', 0 * DACUnits)
    
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
      # [np.linspace(10, 40, 16) * dB, np.linspace(4.91, 4.92, 101) * GHz],
      # save=True, print_data='Amplitude', runs=1)
    
    # run.sweep('Readout Frequency', np.linspace(4.910, 4.925, 151) * GHz,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=1, max_data_dim=1)
    
    # run.value('Qubit Amplitude', 0 * DACUnits)
    
    # run.sweep('Readout Frequency', np.linspace(4.910, 4.925, 151) * GHz,
    # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
    # save=True, runs=1, max_data_dim=1)
    
    # run.sweep(['Qubit Attenuation', 'Readout Frequency'], 
        # [np.linspace(1, 21, 11) * dB, np.linspace(4.91, 4.92, 101) * GHz],
        # save=True, print_data='Amplitude', runs=3)
        
    # run.value('Qubit Amplitude', 0 * DACUnits)
    # run.sweep(['Qubit Attenuation', 'Readout Frequency'], 
        # [np.linspace(1, 21, 11) * dB, np.linspace(4.91, 4.92, 101) * GHz],
        # save=True, print_data='Amplitude', runs=3)
    
    # run.sweep('Qubit T2 Delay', np.linspace(0, 1000, 501) * ns,
        # print_data=['Amplitude'], plot_data=['Q', 'I', 'Amplitude'],
        # save=True, runs=4, max_data_dim=1)
           
    # run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'],
        # [np.linspace(-0.5, 2, 13) * V, np.linspace(3.9, 4.7, 401) * GHz],
        # save=True, print_data='Amplitude', runs=1, max_data_dim=2)
        
    # run.sweep(['Qubit Flux Bias Voltage', 'Qubit Frequency'], 
        # [np.linspace(0.75, 0.95, 3) * V, np.linspace(4.05, 4.25, 101) * GHz],
        # save=True, print_data='Amplitude', runs=1, max_data_dim=2)
        
    # run.sweep('Qubit Frequency', np.linspace(4.55, 4.65, 201) * GHz,
             # print_data=['Q'], plot_data=['Amplitude', 'I', 'Q'],
             # save=True, runs=1, max_data_dim=1)
    

    # run.sweep('Qubit Time', np.linspace(0, 400, 101) * ns,
            # print_data=['Amplitude'], plot_data=['Q', 'I', 'Amplitude'],
            # save=True, runs=1, max_data_dim=1)
            
    # run.sweep(['Qubit Time', 'Qubit Amplitude'], 
        # [np.linspace(0, 1000, 251) * ns, np.linspace(0, .5, 26) * DACUnits],
        # save=True, print_data='Amplitude', runs=2, max_data_dim=2)
        
    # run.sweep(['Qubit Time', 'Qubit Frequency'], 
        # [np.linspace(0, 500, 126) * ns, np.linspace(4.11, 4.17, 121) * GHz],
        # save=True, print_data='Amplitude', runs=2, max_data_dim=2)     
        
    # run.sweep('Qubit Drive to Readout Delay', np.linspace(0, 1000, 251) * ns,
        # print_data=['Amplitude'], plot_data=['Q', 'I', 'Amplitude'],
        # save=True, runs=2, max_data_dim=1)
        
    # run.sweep(['Qubit Attenuation', 'Qubit Frequency'], 
        # [np.linspace(1, 21, 11) * dB, np.linspace(4., 4.2, 101) * GHz],
        # save=True, print_data='Amplitude', runs=2, max_data_dim=2)