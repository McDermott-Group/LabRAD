# HEMT read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import HEMTQubitReadoutExpt as qr


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHzFPGABoards',
                'Boards': [
                            'Shasta Board DAC 9', 
                            'Shasta Board DAC 10',
                            'Shasta Board ADC 6'
                          ],
                'Shasta Board DAC 9':  {
                                        'DAC A': 'Qubit Q',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'Data': False
                                       },
                'Shasta Board DAC 10': {   
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I',
                                       },
                'Shasta Board ADC 6':  {
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
                'Variables': [
                                'Init Time',
                                'Qubit SB Frequency',
                                'Qubit Amplitude',
                                'Qubit Time',
                                'Readout SB Frequency',
                                'Readout Amplitude',
                                'Readout Time',
                                'Qubit Drive to Readout Delay'
                             ]
                },
                # { # GPIB RF Generator
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {'Readout Power': 'Power', 
                    #               'Readout Frequency': 'Frequency'}
                # },
                # { # GPIB RF Generator
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    # 'Variables': {'Qubit Power': 'Power', 
                    #               'Qubit Frequency': 'Frequency'}
                # },
                # { # Lab Brick Attenuator
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7032,
                    # 'Variables': ['Readout Attenuation']
                # },
                # { # Lab Brick Attenuator
                    # 'Interface': 'Lab Brick',
                    # 'Serial Number': 7033,
                    # 'Variables': ['Qubit Attenuation']
                # },
                { # SIM Voltage Source.
                    'Interface': 'SIM928 Voltage Source',
                    'Variables': ['Qubit Flux Bias Voltage']
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Temperature', 
                                  'Reps',
                                  'Rep Iteration'
                                  'Runs',
                                  'ADC Time'],
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
            'Reps': 5000, # should not exceed ~50,000

            'Init Time': 100 * us,

            # 'Qubit Frequency': 20 * GHz,
            # 'Qubit Power': -110 * dBm,
            # 'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0 * DACUnits,
            'Qubit Time': 0 * ns,

            'Qubit Drive to Readout Delay': 0 * ns,

            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 4.991 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 27 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 30 * MHz,
            'Readout Amplitude': 1.0 * DACUnits,
            'Readout Time': 16900 * ns,

            'ADC Wait Time': 500 * ns, # time delay between the start of the readout pulse and the beginning of the demodulation
            
            'Temperature': 16.9 * mK
           }

with qr.HEMTQubitReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.single_shot_iqs(save=False, plot_data=True)
    # run.SingleShotOsc(Save=False, PlotData=['I', 'Q'])
    # run.AvgOsc(Save=True, PlotData=['I', 'Q'], Runs=250)

    # run.sweep('Readout Frequency', np.linspace(3*G, 7*G, 501), PlotData=['I', 'Q', 'ADC Amplitude'], Save=True)
    # run.Sweep('Readout Attenuation', np.linspace(15, 63, 97), PlotData=['I', 'Q', 'ADC Amplitude'], Save=False, MaxDataDim=1)
    
    # run.Sweep(['Readout Attenuation', 'Readout Frequency'], 
              # [np.linspace(1, 61, 21), np.linspace(4.8*G, 4.9*G, 101)], PrintData=['ADC Amplitude'], Save=True)
    
    # run.Sweep(['Flux Bias Voltage', 'Readout Frequency'], 
              # [np.linspace(0, 2, 51), np.linspace(4.83*G, 4.87*G, 201)], PrintData=['ADC Amplitude'], Save=True)