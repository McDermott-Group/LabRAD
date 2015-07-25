# JPM read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import us, ns, V, GHz, MHz, rad, mK, dB, dBm, DACUnits, PreAmpTimeCounts

import JPMQubitReadoutWithResetExpt as qr

Resources = [   { # Waveform parameters.
                    'Resource': 'GHz FPGA Boards', 
                    'Server': 'GHz FPGAs',
                    'Variables': [
                                    'Init Time', 'Bias Time', 'Measure Time',
                                    'Bias Voltage', 'Fast Pulse Time', 'Fast Pulse Amplitude', 'Fast Pulse Width',
                                    'Qubit SB Frequency', 'Qubit Amplitude', 'Qubit Time',
                                    'Readout SB Frequency', 'Readout Amplitude', 'Readout Time', 'Readout Phase',
                                    'Displacement Amplitude', 'Displacement Time', 'Displacement Phase',
                                    'Qubit Drive to Readout',  'Readout to Displacement', 'Displacement to Fast Pulse',
                                    'Readout to Displacement Offset'
                                 ]
                },
                { # DACs are converted to a simple ordered list internally based on 'List Index' value.
                    'Resource': 'DAC',
                    'DAC Name': 'Shasta Board DAC 9',
                    'List Index': 0,
                    'DAC Settings': {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '2.1'
                                    },
                    'Variables': []
                },
                {
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 10',
                    'List Index': 1,
                    'DAC Settings': {   
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I'
                                    },
                    'Variables': []
                },
                # { # GPIB RF Generator.
                    # 'Resource': 'RF Generator',
                    # 'Server': 'GPIB RF Generators',
                    # 'Address': os.environ['COMPUTERNAME'] + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {'Readout Power': 'Power', 
                    #               'Readout Frequency': 'Frequency'}
                # },
                # { # GPIB RF Generator.
                    # 'Resource': 'RF Generator',
                    # 'Server': 'GPIB RF Generators',
                    # 'Address': os.environ['COMPUTERNAME'] + ' GPIB Bus - GPIB0::20::INSTR',
                    # 'Variables': {'Qubit Power': 'Power', 
                                  # 'Qubit Frequency': 'Frequency'}
                # },
                # { # GPIB RF Generator.
                    # 'Resource': 'RF Generator',
                    # 'Server': 'GPIB RF Generators',
                    # 'GPIB Address': 'GPIB0::20',
                    # 'Variables': {'RF Power': 'Power', 
                    #               'RF Frequency': 'Frequency'}
                # },
                # { # Lab Brick Attenuator.
                    # 'Resource': 'Lab Brick Attenuator',
                    # 'Server': os.environ['COMPUTERNAME'] + ' Lab Brick Attenuators',
                    # 'Serial Number': 7032,
                    # 'Variables': {'Readout Attenuation': 'Attenuation'}
                # },
                # { # Lab Brick Attenuator.
                    # 'Resource': 'Lab Brick Attenuator',
                    # 'Server': os.environ['COMPUTERNAME'] + ' Lab Brick Attenuators',
                    # 'Address': 7032,
                    # 'Variables': 'Readout Attenuation'
                # },
                # { # Lab Brick Attenuator.
                    # 'Resource': 'Lab Brick Attenuator',
                    # 'Server': os.environ['COMPUTERNAME'] + ' Lab Brick Attenuators',
                    # 'Address': 7033,
                    # 'Variables': {'Qubit Attenuation': 'Attenuation'}
                # },
                { # SIM Voltage Source.
                    'Resource': 'Voltage Source',
                    'Server': os.environ['COMPUTERNAME'] + ' SIM928',
                    'Address': 'GPIB0::26::SIM900::3',
                    'Variables': ['Qubit Flux Bias Voltage']
                },
                { # External readings.
                    'Resource': 'Manual Record',
                    'Variables': ['Temperature']
                },
                { # Extra experiment parameters.
                    'Resource': 'Software Parameters',
                    'Variables': ['Reps', 'Actual Reps', 'Threshold'],
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
            'Qubit Time': 8000 * ns,
            
            'Qubit Drive to Readout': 0 * ns,
            
            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 20 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 1 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 0 * MHz, 
            'Readout Amplitude': 1 * DACUnits,
            'Readout Time': 1000 * ns,
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
            'Fast Pulse Amplitude': 0.2205 * DACUnits,
            'Fast Pulse Width': 0 * ns,
          
            'Threshold': 100 * PreAmpTimeCounts,
            'Temperature': 14.2 * mK
           }

with qr.JPMQubitReadoutWithReset() as run:
    # run = qr.JPMQubitReadoutWithReset()
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    run.sweep('Fast Pulse Amplitude', np.linspace(0.2, .25, 201) * DACUnits,
            save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])