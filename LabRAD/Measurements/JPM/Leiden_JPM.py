# JPM read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import jpm_qubit_experiments


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Leiden Board DAC 3',
                            'Leiden Board DAC 4'
                          ],
                'Leiden Board DAC 3':  {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Leiden Board DAC 4': {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                       },
                'Variables': {  # Default values.
                                'Init Time': {},
                                'Bias Time': {},
                                'Measure Time': {},
                                'Bias Voltage': {},
                                'Input Bias Voltage': {'Value': 0 * V},
                                'Bias Voltage Step': {'Value': 1 * V},
                                'Bias Voltage Step Time': {'Value': 4 * us},
                                'Max Bias Voltage': {'Value': 1 * V},
                                'Fast Pulse Time': {},
                                'Fast Pulse Amplitude': {},
                                'Fast Pulse Width': {'Value': 0 * ns},
                                'Qubit SB Frequency': {'Value': 0 * MHz},
                                'Qubit Amplitude': {'Value': 0 * DACUnits},
                                'Qubit Time': {'Value': 0 * ns},
                                'Readout SB Frequency': {'Value': 0 * MHz},
                                'Readout Amplitude': {'Value': 0 * DACUnits},
                                'Readout Time': {'Value': 0 * ns},
                                'Readout Phase': {'Value': 0 * rad},
                                'Displacement Amplitude': {'Value': 0 * DACUnits},
                                'Displacement Time': {'Value': 0 * ns},
                                'Displacement Phase': {'Value': 0 * rad},
                                'Qubit Drive to Readout': {'Value': 0 * ns},
                                'Readout to Displacement': {'Value': 0 * ns},
                                'Displacement to Fast Pulse': {'Value': 0 * ns},
                                'Readout to Displacement Offset': {'Value': 0 * DACUnits},
                                'RF SB Frequency': {'Value': 0 * MHz},
                                'RF Amplitude': {'Value': 0 * DACUnits},
                                'RF Time': {'Value': 0 * ns}
                             }
                },
                { # Leiden
                    'Interface': 'Leiden',
                    'Variables': {'Temperature': {'Setting': 'Mix Temperature'}}
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Reps',
                                  'Actual Reps',
                                  'Threshold',
                                  'Preamp Timeout'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH048B-051215A-D10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': r'Z:\mcdermott-group\Data\Matched JPM Photon Counting\Leiden DR 2015-10-02 - Cavity Excitation by JPM',
            'Experiment Name': 'FPA1D',
            'Comments': 'MH048B Qubit and 051215A-D10 JPM in a single box.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 55000, # should not exceed ~50,000
          
            'Init Time': 100 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'Bias Voltage': 0.184 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': .5 * DACUnits,
            'Fast Pulse Width': 0 * ns,
          
            'Threshold': 1000 * PreAmpTimeCounts,
            'Preamp Timeout': 1253 * PreAmpTimeCounts,
           }

with jpm_qubit_experiments.JPMQubitReadout() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    # run.sweep('Bias Voltage', np.linspace(0.18, .21, 301) * V,
        # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])   
    
    run.sweep('Fast Pulse Amplitude', np.linspace(.0, .8, 201) * DACUnits,
          save=False, print_data=['Switching Probability'], plot_data=['Switching Probability'])