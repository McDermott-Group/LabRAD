# HEMT read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
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
                                        'FilterLength': 10000 * ns,
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
                # { # GPIB RF Generator
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    # 'Variables': {  
                                    # 'Qubit Power': {'Setting': 'Power'}, 
                                    # 'Qubit Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': ['Readout Attenuation']
                },
                # { # Lab Brick Attenuator
                    # 'Interface': 'Lab Brick',
                    # 'Serial Number': 7033,
                    # 'Variables': ['Qubit Attenuation']
                # },
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
                                  'Runs',
                                  'ADC Time'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH048B-051215A-D10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Leiden DR 2015-09-25 - Qubits and JPMs',
            'Experiment Name': 'InverseDirectionCavitySpectroscopy',
            'Comments': 'Driving qubit cavity through JPM Fast Pulse line.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 5000, # should not exceed ~50,000

            'Init Time': 100 * us,

            # 'Qubit Frequency': 20 * GHz,
            # 'Qubit Power': -110 * dBm,
            # 'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            # 'Qubit SB Frequency': 0 * MHz,
            # 'Qubit Amplitude': 0 * DACUnits,
            # 'Qubit Time': 0 * ns,

            'Qubit Drive to Readout Delay': 0 * ns,

            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 4.821 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 1 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 0 * MHz,
            'Readout Amplitude': 1.0 * DACUnits,
            'Readout Time': 17400 * ns,

            'ADC Wait Time': 1000 * ns, # time delay between the start of the readout pulse and the start of the demodulation
           }

with hemt_qubit_experiments.HEMTQubitReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)

    # run.single_shot_iqs(save=False, plot_data=True)
    # run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=True, plot_data=['I', 'Q'], runs=250)

    run.sweep('Readout Frequency', np.linspace(4.65, 4.85, 201) * GHz,
              plot_data=['I', 'Q', 'ADC Amplitude'], save=True)