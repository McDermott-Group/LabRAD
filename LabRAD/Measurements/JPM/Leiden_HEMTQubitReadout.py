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
                'Leiden Board DAC 3':  {
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
                                        'FilterWidth': 9500 * ns,
                                        'FilterLength': 8000 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -30 * MHz,
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
                                'Stark Amplitude': {'Value': 0 * DACUnits},
                                'Stark Time': {'Value': 0 * ns}
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
                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    'Variables': {  
                                    'Qubit Power': {'Setting': 'Power'}, 
                                    'Qubit Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7031,
                    'Variables': ['Readout Attenuation']
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': ['Qubit Attenuation']
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
                    'Variables': ['Reps',
                                  'Runs'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH060',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'StarkShift2D',
            'Comments': 'Driving with IQ modulation, test of Hittite HMC451 amplifier on DAC IQ mixer' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 4000, # should not exceed ~5,000, use agrument "runs" in sweep parameters instead 

            'Init Time': 100 * us,
            
            'Stark Amplitude': 0 * DACUnits,
            'Stark Time': 10000 * ns,

            'Readout Frequency': 4.9179 * GHz, #4.919 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 30 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 30 * MHz,
            'Readout Amplitude': 0.5 * DACUnits,
            'Readout Time': 2000 * ns,
            
            'Qubit Drive to Readout Delay': 10 * ns,
            
            'Qubit Frequency': 4.758 * GHz,
            'Qubit Power': 13 * dBm,
            'Qubit Attenuation': 1 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0.5 * DACUnits,
            'Qubit Time': 2000 * ns,

            'Qubit Flux Bias Voltage': -0.15 * V,
            
            'ADC Wait Time': 0 * ns, # time delay between the start of the readout pulse and the start of the demodulation
           }

#with hemt_qubit_experiments.HEMTQubitReadout() as run:
with hemt_qubit_experiments.HEMTStarkShift() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)

    # run.single_shot_iqs(save=True, plot_data=True)
    # run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=True, plot_data=['I', 'Q'], runs=200)

    # run.sweep('Readout Frequency', np.linspace(4.910, 4.925, 301) * GHz,
            # plot_data=['Amplitude','I','Q'], save=True, runs=1, max_data_dim=1)
            
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
      # [np.linspace(9, 29, 21) * dB, np.linspace(4.91, 4.925, 151) * GHz],
      # save=True, print_data='Amplitude', runs=2)
    
    # run.value('Qubit Amplitude', 0.75)
    # run.value('Qubit Time', 20)
       
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
        # [np.linspace(1, 40, 40) * dB, np.linspace(4.91, 4.925, 151) * GHz],
        # save=True, print_data='Amplitude', runs=1)
        
    # run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'],
        # [np.linspace(-1.5, 1.5, 31) * V, np.linspace(4.91, 4.925, 76) * GHz],
        # save=True, print_data='Amplitude', runs=1, max_data_dim=2)
        
    # run.sweep(['Qubit Flux Bias Voltage', 'Qubit Frequency'], 
        # [np.linspace(-0.25, .75, 26) * V, np.linspace(4.25, 4.9, 651) * GHz],
        # save=True, print_data='Amplitude', runs=1, max_data_dim=2)
        
    # run.sweep('Qubit Frequency', np.linspace(4.3, 4.7, 801) * GHz,
             # print_data=['Amplitude'], plot_data=['Amplitude', 'I', 'Q'],
             # save=True, runs=1, max_data_dim=1)
             
    # run.sweep('Qubit Time', np.linspace(0,250,251) * ns,
            # print_data=['Amplitude'], plot_data=['Amplitude', 'I', 'Q'],
            # save = True, runs=1, max_data_dim=1)
        
    run.sweep(['Stark Amplitude', 'Qubit Frequency'], 
        [np.linspace(0, .5, 26) * DACUnits, np.linspace(4.3, 4.8, 501) * GHz],
        save=True, print_data='Amplitude', runs=2, max_data_dim=2)