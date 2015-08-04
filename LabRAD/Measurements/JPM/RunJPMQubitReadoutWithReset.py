# JPM read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import JPMQubitReadoutWithResetExpt as qr


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Shasta Board DAC 9', 
                            'Shasta Board DAC 10',
                            # 'Shasta Board ADC 11'
                          ],
                'Shasta Board DAC 9':  {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '1.0',
                                        'Data': True
                                       },
                'Shasta Board DAC 10': {   
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'Readout I',
                                       },
                'Shasta Board ADC 11':  {
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
                                       },
                'Variables': [
                                'Init Time',
                                'Bias Time', 
                                'Measure Time',
                                'Bias Voltage', 
                                'Fast Pulse Time', 
                                'Fast Pulse Amplitude', 
                                'Fast Pulse Width',
                                'Qubit SB Frequency', 
                                'Qubit Amplitude',
                                'Qubit Time',
                                'Readout SB Frequency',
                                'Readout Amplitude',
                                'Readout Time',
                                'Readout Phase',
                                'Displacement Amplitude',
                                'Displacement Time',
                                'Displacement Phase',
                                'Qubit Drive to Readout',
                                'Readout to Displacement',
                                'Displacement to Fast Pulse',
                                'Readout to Displacement Offset'
                             ]
                },
                # { # GPIB RF Generator.
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {'Readout Power': 'Power', 
                    #               'Readout Frequency': 'Frequency'}
                # },
                # { # GPIB RF Generator.
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {'Qubit Power': 'Power', 
                    #               'Qubit Frequency': 'Frequency'}
                # },
                # { # GPIB RF Generator.
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    # 'Variables': {'RF Power': 'Power', 
                    #               'RF Frequency': 'Frequency'}
                # },
                # { # Lab Brick Attenuator.
                    # 'Interface': 'Lab Brick Attenuator'
                    # 'Server': 'LabBrickAttenuator',
                    # 'Address': 7032,
                    # 'Variables': ['Readout Attenuation']
                # },
                # { # Lab Brick Attenuator.
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7032,
                    # 'Variables': ['Readout Attenuation']
                # },
                # { # Lab Brick Attenuator.
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7033,
                    # 'Variables': ['Qubit Attenuation']
                # },
                { # SIM Voltage Source.
                    'Interface': 'SIM928 Voltage Source',
                    'Variables': 'Qubit Flux Bias Voltage'
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Temperature', 
                                  'Reps',
                                  'Actual Reps',
                                  'Threshold'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'Test Device',
            'User': 'Test User',
            'Base Path': 'Z:\mcdermott-group\Data\Test',
            'Experiment Name': 'SimpleExperiment',
            'Comments': '' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 3000, # should not exceed ~50,000
          
            'Qubit Frequency': 20 * GHz,
            'Qubit Power': -110 * dBm, 
            'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0.5 * DACUnits,
            'Qubit Time': 60 * ns,
            
            'Qubit Drive to Readout': 0 * ns,
            
            'Qubit Flux Bias Voltage': 0.28 * V,

            'Readout Frequency': 20 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 1 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 0 * MHz, 
            'Readout Amplitude': 1 * DACUnits,
            'Readout Time': 50 * ns,
            'Readout Phase': 0 * rad,
            
            'Readout to Displacement': 0 * ns,
            'Readout to Displacement Offset': 0.00 * DACUnits,
            
            'Displacement Amplitude': 0.0 * DACUnits,
            'Displacement Time': 0 * ns,
            'Displacement Phase': 0 * rad,
            
            'Displacement to Fast Pulse': -100 * ns,  # time delay between the end of the displacement pulse and the start of the fast pulse
          
            'Init Time': 500 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'Bias Voltage': 0.194 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': 1.0 * DACUnits,
            'Fast Pulse Width': 0 * ns,
          
            'Threshold': 100 * PreAmpTimeCounts,
            'Temperature': 14.2 * mK
           }

with qr.JPMQubitReadoutWithReset() as run:
    # run = qr.JPMQubitReadoutWithReset()
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    run.sweep('Fast Pulse Amplitude', np.linspace(0.5, 1, 501) * DACUnits,
            save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.sweep('Qubit Flux Bias Voltage', np.linspace(0, 1, 1001) * V,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])