# HEMT read out of a qubit connected to a resonator.

import sys
import numpy as np

import labrad as lr

import HEMTQubitReadoutExpt as qr

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
                                    'Init Time',
                                    'Qubit SB Frequency', 'Qubit Amplitude', 'Qubit Time',
                                    'Readout SB Frequency', 'Readout Amplitude', 'Readout Time',
                                    'Qubit Drive to Readout Delay'
                                 ]
                },
                { # DACs are converted to a simple ordered list internally based on 'List Index' value.
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 3',
                    'List Index': 0,
                    'DAC Settings': {
                                        'DAC A': 'None',
                                        'DAC B': 'Readout I',
                                    },
                    'Variables': []
                },
                {
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 4',
                    'List Index': 1,
                    'DAC Settings': {   
                                        'DAC A': 'Qubit Q',
                                        'DAC B': 'Qubit I'
                                    },
                    'Variables': []
                },
                { # ADCs are converted to a simple ordered list internally based on 'List Index' value.  
                    'Resource': 'ADC',
                    'ADC Name': 'Leiden Board ADC 5',
                    'List Index': 0,
                    'ADC Settings': { # These default settings might be over-written by the Experiment methods. 
                                        'RunMode': 'demodulate', #'average'
                                        'FilterType': 'square',
                                        'FilterWidth': 9500,
                                        'FilterLength': 10000,
                                        'filterStretchAt': 0,
                                        'filterStretchLen': 0,
                                        'DemodPhase': 0,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -30*M,
                                        'ADCDelay': 0
                                    },
                    'Variables': ['ADC Wait Time'],
                },
                { # GPIB RF Generator
                    'Resource': 'RF Generator',
                    'GPIB Address': 'GPIB0::19',
                    'Variables': ['Readout Power', 'Readout Frequency']
                },
                # { # GPIB RF Generator
                    # 'Resource': 'RF Generator',
                    # 'GPIB Address': 'GPIB0::19',
                    # 'Variables': ['Qubit Power', 'Qubit Frequency']
                # },
                { # Lab Brick Attenuator
                    'Resource': 'Lab Brick',
                    'Serial Number': 7033,
                    'Variables': ['Readout Attenuation']
                },
                # { # Lab Brick Attenuator
                    # 'Resource': 'Lab Brick',
                    # 'Serial Number': 7032,
                    # 'Variables': ['Qubit Attenuation']
                # },
                { # SIM Voltage Source
                    'Resource': 'SIM',
                    'GPIB Address': 'GPIB::25',
                    'SIM Slot': 7,
                    'Variables': ['Flux Bias Voltage']
                },
                { # External readings.
                    'Resource': 'Manual Record',
                    'Variables': ['Temperature']
                },
                { # Extra experiment parameters.
                    'Resource': 'Software Parameters',
                    'Variables': ['Reps', 'Runs', 'ADC Time', 'Rep Iteration'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MWO72',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits',
            'Experiment Name': 'ROAttn1D_Calib',
            'Comments': 'The calibration line is connected to the cold HEMT via the cold relays.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 15000, # should not exceed ~50,000

            'Init Time': 100, # us

            # 'Qubit Frequency': 20*G, # Hz
            # 'Qubit Power': -110, # dBm
            # 'Qubit Attenuation': 63, # dB, should be in (0, 63] range
            'Qubit SB Frequency': 0*M, # Hz
            'Qubit Amplitude': 0, # DAC units
            'Qubit Time': 0, # ns

            'Qubit Drive to Readout Delay': 0, # ns

            'Flux Bias Voltage': 0, # V

            'Readout Frequency': 4.991*G, # Hz
            'Readout Power': 13, # dBm
            'Readout Attenuation': 63, # dB, should be in (0, 63] range
            'Readout SB Frequency': 30*M, # Hz
            'Readout Amplitude': 0.5, # DAC units
            'Readout Time': 16900, # ns

            'ADC Wait Time': 500, # ns (time delay between the start of the readout pulse and the beginning of the demodulation).
            
            'Temperature': 11.9 # mK
           }

with qr.HEMTQubitReadout(cxn) as run:
    
    run.SetExperiment(ExptInfo, Resources, ExptVars)
    
    # run.SingleShotIQs(Save=False, PlotData=True)
    # run.SingleShotOsc(Save=False, PlotData=['I', 'Q'])
    # run.AvgOsc(Save=True, PlotData=['I', 'Q'], Runs=250)
    
    run.ChangeADCSetting('Leiden Board ADC 5', 'RunMode', 'demodulate')
    
    # run.Sweep('Readout Frequency', np.linspace(4.85*G, 5*G, 151), PlotData=['I', 'Q', 'ADC Amplitude'], Save=True)
    run.Sweep('Readout Attenuation', np.linspace(15, 63, 97), PlotData=['I', 'Q', 'ADC Amplitude'], Save=False, MaxDataDim=1)
    
    # run.Sweep(['Readout Attenuation', 'Readout Frequency'], 
              # [np.linspace(1, 51, 26), np.linspace(4.97*G, 5.01*G, 101)], PrintData=['I', 'Q'], Save=True)