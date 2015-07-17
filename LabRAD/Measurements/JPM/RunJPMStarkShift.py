# Stark shift experiment with a JPM.

import sys
import numpy as np

import labrad as lr

import JPMStarkShiftExpt as ss

G = float(10**9)
M = float(10**6)

# Connect to LabRAD.
cxn = lr.connect()

# List of the experiment resources. Simply uncomment/comment the devices that should be used/unused.
# However, 'Resource': 'LabRAD Server' should never be left out.
Resources = [   { # This is a required resource.
                    'Resource': 'LabRAD Server', 
                    'Server Name': 'ghz_fpgas'
                },
                { # List here all parameters that specify waveforms.
                    'Resource': 'Waveform Parameters',
                    'Variables': [
                                    'Init Time', 'Bias Time', 'Measure Time',
                                    'Bias Voltage', 'Fast Pulse Time', 'Fast Pulse Amplitude', 'Fast Pulse Width',
                                    'Qubit SB Frequency', 'Qubit Amplitude', 'Qubit Time',
                                    'Readout SB Frequency', 'Readout Amplitude', 'Readout Time',
                                    'Stark Amplitude', 'Stark Time',
                                    'Qubit Drive to Readout Delay', 'Readout to Fast Pulse Delay'
                                 ]
                },
                { # DACs are converted to a simple ordered list internally based on 'List Index' value.
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 3',
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
                    'DAC Name': 'Leiden Board DAC 4',
                    'List Index': 1,
                    'DAC Settings': {   
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I'
                                    },
                    'Variables': []
                },
                { # GPIB RF Generator
                    'Resource': 'RF Generator',
                    'GPIB Address': 'GPIB0::19',
                    'Variables': ['Readout Power', 'Readout Frequency']
                },
                { # GPIB RF Generator
                    'Resource': 'RF Generator',
                    'GPIB Address': 'GPIB0::20',
                    'Variables': ['Qubit Power', 'Qubit Frequency']
                },
                { # Lab Brick Attenuator
                    'Resource': 'Lab Brick',
                    'Serial Number': 7032,
                    'Variables': ['Readout Attenuation']
                },
                { # Lab Brick Attenuator
                    'Resource': 'Lab Brick',
                    'Serial Number': 7033,
                    'Variables': ['Qubit Attenuation']
                },
                { # SIM Voltage Source
                    'Resource': 'SIM',
                    'GPIB Address': 'GPIB::26',
                    'SIM Slot': 3,
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
            'Device Name': 'MW072 - JPM 051215A-D10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits',
            'Experiment Name': 'StarkShift2D',
            'Comments': 'RF generators - DACs - attenuators - 3 dB splitter - 50 dB - LP filter - qubit - LP filter - relay - isolator - relay - JPM.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 6000, # should not exceed ~50,000
            
            'Stark Amplitude': 0, # DAC units
            'Stark Time': 10000, # ns
            
            'Qubit Frequency': 4.832*G, #4.532*G, # Hz
            'Qubit Power': 13, # dBm
            'Qubit Attenuation': 1, # dB, should be in (0, 63] range
            'Qubit SB Frequency': 0*M, # Hz
            'Qubit Amplitude': 0.5, # DAC units
            'Qubit Time': 8000, # ns
            
            'Qubit Drive to Readout Delay': 0, # ns
            
            'Qubit Flux Bias Voltage': 0.28, # V

            'Readout Frequency': 4.9945*G, # Hz
            'Readout Power': 13, # dBm
            'Readout Attenuation': 45, # dB, should be in (0, 63] range
            'Readout SB Frequency': 30*M, # Hz
            'Readout Amplitude': 0.5, # DAC units
            'Readout Time': 500, # ns
          
            'Readout to Fast Pulse Delay': -100, # ns (time delay between the end of the readout pulse and the start of the fast pulse).
          
            'Init Time': 500, # microseconds
            'Bias Time': 100, # microseconds
            'Measure Time': 50, # microseconds
          
            'Bias Voltage': 0.199, # FastBias DAC units
            'Fast Pulse Time': 3, # nanoseconds
            'Fast Pulse Amplitude': 0.34, # DAC units
            'Fast Pulse Width': 0, #nanoseconds
          
            'Threshold': 100,   # Preamp Time Counts
            'Temperature': 11.5 # mK
           }

with ss.JPMStarkShift(cxn) as run:
    
    run.SetExperiment(ExptInfo, Resources, ExptVars)
        
    # run.Sweep(['Qubit Frequency', 'Stark Amplitude'], [np.linspace(4.4*G, 4.9*G, 251), np.linspace(0, 0.25, 26)],
        # Save=True, PrintData=['Switching Probability'])
        
    run.Sweep(['Qubit Frequency', 'Stark Amplitude'], [np.linspace(4.4*G, 4.6*G, 201), np.linspace(0, 0.25, 26)],
        Save=True, PrintData=['Switching Probability'])