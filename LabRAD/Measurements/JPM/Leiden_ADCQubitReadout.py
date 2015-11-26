# HEMT read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import adc_qubit_experiments


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Leiden Board DAC 3', 
                            'Leiden Board DAC 4',
                            'Leiden Board ADC 5'
                          ],
                'Leiden Board DAC 3': {
                                        'DAC A': 'None',
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
                                        'FilterType': 'square',
                                        'FilterWidth': 1000 * ns,
                                        'FilterStartAt': 0 * ns,
                                        'FilterLength': 100 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -200 * MHz,
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
                                # 'Qubit T2 Delay': {'Value': 0 * ns},
                                'ADC Demod Frequency': {'Value': -50 * MHz}
                             }
                },
                # { # GPIB RF Generator
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {  
                                    # 'Qubit Power': {'Setting': 'Power'}, 
                                    # 'Qubit Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {  
                                    'Qubit Power': {'Setting': 'Power'}, 
                                    'Qubit Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab RF Generator
                    'Interface': 'Lab Brick RF Generator',
                    'Serial Number': 10776,
                    'Variables': {  
                                    'Readout Power': {'Setting': 'Power'}, 
                                    'Readout Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7031,
                    'Variables': 'Qubit Attenuation'
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': 'Readout Attenuation'
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
            'Device Name': 'MH060',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'ADCOptimizations',
            'Comments': 'ADC band pass filters removed. DC blocks and filters on Lab Brick attenuators. Updated filter functions.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 4000, # should not exceed ~5,000, use argument "runs" in sweep parameters instead 

            'Init Time': 100 * us,
            
            # 'Stark Amplitude': 0 * DACUnits,
            # 'Stark Time': 10000 * ns,

            'Readout Frequency': 4.9134 * GHz,
            'Readout Power': 10 * dBm,
            'Readout Attenuation': 1 * dB, #42 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 125 * MHz, # no filters on the IQ-mixer
            'Readout Amplitude': 1.0 * DACUnits,
            'Readout Time': 100 * ns,
            
            'Qubit Drive to Readout Delay': 0 * ns,
            
            'Qubit Frequency': 4.5827 * GHz,
            'Qubit Power': 13 * dBm,
            'Qubit Attenuation': 11 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 62.5 * MHz,
            'Qubit Amplitude': 0.0 * DACUnits, #1.0 * DACUnits,
            'Qubit Time': 5000 * ns, #60 * ns,
            # 'Qubit T2 Delay': 0 * ns,

            'Qubit Flux Bias Voltage': 0 * V,
                        
            'ADC Wait Time': 0 * ns,
            # 'ADC Demod Frequency': -50 * MHz
           }

# with adc_qubit_experiments.ADCStarkShift() as run:
# with adc_qubit_experiments.ADCRamsey() as run:
with adc_qubit_experiments.ADCQubitReadout() as run:
# with adc_qubit_experiments.ADCDemodTest() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    # run.single_shot_iqs(save=True, plot_data=True)
    # run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=True, plot_data=['I', 'Q'], runs=1000)

    # run.sweep('Readout Attenuation', np.linspace(10, 40, 61) * dB,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=1, max_data_dim=1)
        
    # run.value('Readout Amplitude', 1 * DACUnits)
    # run.sweep('Readout Attenuation', np.linspace(30, 63, 23) * dB,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=5, max_data_dim=1)
        
    # run.value('Readout Amplitude', 0 * DACUnits)
    # run.sweep('Readout Attenuation', np.linspace(30, 63, 23) * dB,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=5, max_data_dim=1)
    
    # run.value('Qubit Amplitude', 1 * DACUnits)
    # run.sweep('Qubit Attenuation', np.linspace(1, 40, 40) * dB,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=4, max_data_dim=1)
        
    # run.value('Qubit Amplitude', 0 * DACUnits)
    # run.sweep('Qubit Attenuation', np.linspace(1, 40, 40) * dB,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=4, max_data_dim=1)
        
    # run.value('Qubit Amplitude', 1 * DACUnits)
    # run.sweep('Readout Frequency', np.linspace(4.91, 4.925, 151) * GHz,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=5, max_data_dim=1)
        
    # run.value('Qubit Amplitude', 0 * DACUnits)
    # run.sweep('Readout Frequency', np.linspace(4.91, 4.925, 151) * GHz,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=5, max_data_dim=1)
    
    # for freq in np.linspace(4.918, 4.920, 11) * GHz:
        # run.value('Readout Frequency', freq)
    # for ampl in np.linspace(0, 1, 6) * DACUnits:
        # run.value('Readout Amplitude', ampl)
    # for attn in np.linspace(25, 45, 21) * dB:
        # run.value('Readout Attenuation', attn)
    # run.sweep('Readout Time', np.linspace(0, 400, 101) * ns,
        # print_data = ['Amplitude', 'Temperature'],
        # plot_data = ['Amplitude', 'I', 'Q'],
        # save=True, runs=2, max_data_dim=2)
    
    run.sweep('ADC Wait Time', np.linspace(-700, 500, 201) * ns,
        print_data = ['Amplitude', 'Temperature'],
        plot_data = ['Amplitude', 'I', 'Q'],
        save=True, runs=4, max_data_dim=1)
    
    # run.sweep(['Qubit Frequency', 'Readout Frequency'], 
      # [np.linspace(4, 5, 101) * GHz, np.linspace(4., 5, 101) * GHz],
      # save=True, print_data='Amplitude', runs=1)
    
    # run.value('Qubit Amplitude', 1 * DACUnits)
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
      # [np.linspace(1, 51, 26) * dB, np.linspace(4.91, 4.925, 76) * GHz],
      # save=True, print_data='Amplitude', runs=4)
      
    # run.value('Qubit Amplitude', 0 * DACUnits)
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
      # [np.linspace(1, 51, 51) * dB, np.linspace(4.91, 4.925, 76) * GHz],
      # save=True, print_data='Amplitude', runs=5)
      
    # run.sweep(['Readout Time', 'Readout Frequency'], 
      # [np.linspace(0, 4000, 251) * ns, np.linspace(4.91, 4.925, 76) * GHz],
      # save=True, print_data='Amplitude', runs=1)
    
    # run.sweep('Readout Time', np.linspace(0, 800, 201) * ns,
        # save=True, print_data='Amplitude', plot_data=['I','Q','Amplitude'], runs=1, max_data_dim=2)
        
    # run.value('Readout Frequency', 4.9144 * GHz)
    
    # run.sweep('Readout Time', np.linspace(0, 800, 201) * ns,
        # save=True, print_data='Amplitude',plot_data=['I','Q','Amplitude'], runs=1, max_data_dim=2)
    
    # run.sweep('Readout Frequency', np.linspace(4.910, 4.915, 51) * GHz,
        # print_data=['Amplitude','I','Q'], plot_data=['Amplitude','I','Q'],
        # save=True, runs=1, max_data_dim=1)
    
    # run.sweep(['Readout Time', 'Readout Frequency'], 
      # [np.linspace(0, 200, 51) * ns, np.linspace(4.91, 4.925, 76) * GHz],
      # save=True, print_data='Amplitude', runs=2)
    
    # run.value('Qubit Time', 60 * ns)
    # run.value('Qubit Amplitude', 1 * DACUnits)
    # run.sweep(['Readout Time', 'Readout Frequency'], 
      # [np.linspace(0, 200, 51) * ns, np.linspace(4.91, 4.925, 76) * GHz],
      # save=True, print_data='Amplitude', runs=2)
    
    # run.sweep(['Qubit Attenuation', 'Readout Frequency'], 
        # [np.linspace(1, 21, 11) * dB, np.linspace(4.91, 4.92, 101) * GHz],
        # save=True, print_data='Amplitude', runs=3)
        
    # run.value('Qubit Amplitude', 0 * DACUnits)
    # run.sweep(['Qubit Attenuation', 'Readout Frequency'], 
        # [np.linspace(1, 21, 11) * dB, np.linspace(4.91, 4.92, 101) * GHz],
        # save=True, print_data='Amplitude', runs=3)
    
    # run.sweep('Qubit T2 Delay', np.linspace(0, 1000, 251) * ns,
        # print_data=['Amplitude'], plot_data=['Q', 'I', 'Amplitude'],
        # save=True, runs=4, max_data_dim=1)
           
    # run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'],
        # [np.linspace(-0.5, 2, 13) * V, np.linspace(3.9, 4.7, 401) * GHz],
        # save=True, print_data='Amplitude', runs=1, max_data_dim=2)
        
    # run.sweep(['Qubit Flux Bias Voltage', 'Qubit Frequency'], 
        # [np.linspace(0., 2, 11) * V, np.linspace(3.9, 4.6, 301) * GHz],
        # save=True, print_data='Amplitude', runs=1, max_data_dim=2)
        
    # run.sweep('Qubit Frequency', np.linspace(4.50, 4.65, 301) * GHz,
             # print_data=['I', 'Q'], plot_data=['I', 'Q', 'Amplitude'],
             # save=True, runs=10, max_data_dim=1)

    # run.sweep('Qubit Time', np.linspace(0, 2000, 501) * ns,
            # print_data=['Amplitude', 'Temperature'], 
            # plot_data=['Q', 'I', 'Amplitude'],
            # save=True, runs=10, max_data_dim=1)
            
    # run.sweep(['Qubit Time', 'Qubit Amplitude'], 
        # [np.linspace(0, 1000, 251) * ns, np.linspace(0, 1, 21) * DACUnits],
        # save=True, print_data='Amplitude', runs=2, max_data_dim=2)
        
    # run.sweep(['Qubit Time', 'Qubit Frequency'], 
        # [np.linspace(0, 500, 126) * ns, np.linspace(4.11, 4.17, 121) * GHz],
        # save=True, print_data='Amplitude', runs=2, max_data_dim=2)     
        
    # run.sweep('Qubit Drive to Readout Delay', np.linspace(0, 10000, 2501) * ns,
        # print_data=['Amplitude'], plot_data=['Q', 'I', 'Amplitude'],
        # save=True, runs=5, max_data_dim=1)
        
    # run.sweep(['Qubit Attenuation', 'Qubit Frequency'], 
        # [np.linspace(1, 21, 11) * dB, np.linspace(4.55, 4.6, 125) * GHz],
        # save=True, print_data='Amplitude', runs=2, max_data_dim=2)