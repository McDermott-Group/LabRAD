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
                                        'DAC B': 'None',
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
                                'ADC Wait Time': {'Value': 0 * ns}
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
                                  'Rep Iteration'
                                  'Runs'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH060',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'FluxBiasROFreq2D',
            'Comments': 'Readout frequency vs. readout power for MH060 qubit. Driving with IQ modulation.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 1000, # should not exceed ~5,000, use agrument "runs" in sweep parameters instead 

            'Init Time': 100 * us,

            'Qubit Frequency': 20 * GHz,
            'Qubit Power': -110 * dBm,
            'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0 * DACUnits,
            'Qubit Time': 0 * ns,

            'Qubit Drive to Readout Delay': 10 * ns,

            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 4.913 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 24 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 30 * MHz,
            'Readout Amplitude': 0.5 * DACUnits,
            'Readout Time': 3000 * ns,

            'ADC Wait Time': 0 * ns, # time delay between the start of the readout pulse and the start of the demodulation
           }


with hemt_qubit_experiments.HEMTQubitReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)

    # run.single_shot_iqs(save=False, plot_data=True)
    # run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=False, plot_data=['I', 'Q'], runs=30)

    # run.sweep('Readout Frequency', np.linspace(4.85, 4.95, 101) * GHz,
             # plot_data=['Amplitude'], save=True, runs=3)
              
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
        # [np.linspace(1, 41, 21) * dB, np.linspace(4.9, 4.93, 151) * GHz],
        # save=True, print_data='Amplitude')
        
    run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'], 
        [np.linspace(-1.5, 1.5, 31) * V, np.linspace(4.908, 4.923, 101) * GHz],
        save=True, print_data='Amplitude', runs=25, max_data_dim=2)