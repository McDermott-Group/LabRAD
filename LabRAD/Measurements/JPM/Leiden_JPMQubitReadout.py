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
                                        'DAC B': 'Readout I',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Leiden Board DAC 4': {
                                        'DAC A': 'None',
                                        'DAC B': 'Readout I',
                                       },
                'Variables': {  # Default values.
                                'Init Time': {},
                                'Bias Time': {},
                                'Measure Time': {},
                                'Bias Voltage': {},
                                # 'Input Bias Voltage': {'Value': 0 * V},
                                # 'Bias Voltage Step': {'Value': 1 * V},
                                # 'Bias Voltage Step Time': {'Value': 4 * us},
                                # 'Max Bias Voltage': {'Value': 1 * V},
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
                                # 'RF SB Frequency': {'Value': 0 * MHz},
                                # 'RF Amplitude': {'Value': 0 * DACUnits},
                                # 'RF Time': {'Value': 0 * ns},
                                # 'Readout to Fast Pulse': {'Value': 0 * ns},
                                # 'Stark Amplitude': {'Value': 0 * DACUnits},
                                # 'Stark Time': {'Value': 0 * ns}
                             }
                },
                # { # GPIB RF Generator.
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    # 'Variables': {
                                    # 'RF Power': {'Setting': 'Power'}, 
                                    # 'RF Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
                { # GPIB RF Generator.
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    'Variables': {
                                    'Readout Power': {'Setting': 'Power'}, 
                                    'Readout Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # GPIB RF Generator.
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {
                                    'Qubit Power': {'Setting': 'Power'}, 
                                    'Qubit Frequency': {'Setting': 'Frequency'}
                                 }
                },
                # { # Lab Brick Attenuator.
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7033,
                    # 'Variables': ['RF Attenuation']
                # },
                { # Lab Brick Attenuator.
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': ['Readout Attenuation']
                },
                { # Lab Brick Attenuator.
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7031,
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
                                  'Actual Reps',
                                  'Threshold',
                                  'Preamp Timeout'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH061B-051215A-D9',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'ROAttnFreq2D',
            'Comments': 'Qubit MH061B and JPM 051215A-D9 in a single box.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 1000, # should not exceed ~55,000
          
            # 'Stark Amplitude': 1 * DACUnits,
            # 'Stark Time': 10 * us,
            # 'Readout to Fast Pulse': 0 * ns,
          
            'Qubit Frequency': 20 * GHz,
            'Qubit Power': -110 * dBm, 
            'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0.0 * DACUnits,
            'Qubit Time': 0 * ns,
            
            'Qubit Drive to Readout': 0 * ns,
            
            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 20 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 5 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 0 * MHz, 
            'Readout Amplitude': 0.5 * DACUnits,
            'Readout Time': 500 * ns,
            'Readout Phase': 0 * rad,
            
            'Readout to Displacement': 0 * ns,
            'Readout to Displacement Offset': 0.00 * DACUnits,
            
            'Displacement Amplitude': 0.0 * DACUnits,
            'Displacement Time': 0 * ns,
            'Displacement Phase': 0 * rad,
            
            'Displacement to Fast Pulse': -10 * ns,  # time delay between the end of the displacement pulse and the start of the fast pulse
          
            'Init Time': 1000 * us,
            'Bias Time': 100 * us,
            'Measure Time': 10 * us,
          
            'Bias Voltage': 0.191 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': .463 * DACUnits, #.486 * DACUnits,
            'Fast Pulse Width': 0 * ns,
          
            'Preamp Timeout': 253 * PreAmpTimeCounts, # FB15, PA5: 3.3, 22, 33820
            'Threshold': 253 * PreAmpTimeCounts
           }

with jpm_qubit_experiments.JPMQubitReadout() as run:
# with jpm_qubit_experiments.JPMStarkShift() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    # run.sweep('Bias Voltage', np.linspace(0.188, .2, 101) * V,
        # save=False, print_data=['Switching Probability'], plot_data=['Switching Probability'])   
    
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.45, .5, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
          
    # run.sweep('Init Time', np.array([20, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 3000, 5000]) * us,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
          
    # run.sweep('Readout Frequency', np.linspace(4.4, 5.1, 351) * GHz,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    
    run.sweep(['Readout Attenuation', 'Readout Frequency'], 
            [np.linspace(10, 40, 11) * dB, np.linspace(4.4, 5.1, 351) * GHz],
            save=True, print_data=['Switching Probability'])
            
    # run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'], 
        # [np.linspace(-3, 3, 15) * V, np.linspace(4.81, 4.86, 101) * GHz],
        # save=True, print_data=['Switching Probability'])
    
    # run.sweep('Qubit Flux Bias Voltage', np.linspace(0, 1, 1001) * V,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])