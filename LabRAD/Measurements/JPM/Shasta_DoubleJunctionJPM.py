# JPM read out of a qubit connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import jpm_qubit_experiments


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Shasta Board DAC 9', 
                            'Shasta Board DAC 10',
                          ],
                'Shasta Board DAC 9':  {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Shasta Board DAC 10': {   
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                       },
                'Variables': {  # Default values.
                                'Init Time': {},
                                'Bias Time': {},
                                'Measure Time': {},
                                'Bias Voltage': {},
                                'Input Bias Voltage': {'Value': 0 * V},
                                'Bias Voltage Step': {'Value': .2 * V},
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
                { # GPIB RF Generator.
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    'Variables': {
                                    'RF Power': {'Setting': 'Power'}, 
                                    'RF Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator.
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7031,
                    'Variables': 'RF Attenuation'
                },
                # { # SIM Voltage Source.
                    # 'Interface': 'SIM928 Voltage Source',
                    # 'Address': ('SIM900 - ' + comp_name + 
                                # ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    # 'Variables': 'Qubit Flux Bias Voltage'
                # },
                { # ADR3
                    'Interface': 'ADR3',
                    'Variables': {
                                    'Temperature': {'Setting': 'Temperatures',
                                                    'Stage': 'FAA'}
                                 }
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': [
                                  'Reps',
                                  'Actual Reps',
                                  'Threshold'
                                 ],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': '080415A-G4',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Matched JPM Photon Counting\ADR3 2015-08-26 - Double JPM 080415A-G4',
            'Experiment Name': 'BV2D',
            'Comments': 'Extra 10 dB on top of the fridge(?)' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 60, # should not exceed ~50,000

            'RF Frequency': 20 * GHz,
            'RF Power': 13 * dBm,
            'RF Attenuation': 63 * dB, # should be in (0, 63] range
   
            'Init Time': 500 * us,
            'Bias Time': 100 * us,
            'Measure Time': 30 * us,
          
            'Input Bias Voltage': 0.45 * V,
            
            'Bias Voltage': -.435 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': .5 * DACUnits,

            'Threshold': 500 * PreAmpTimeCounts,
           }

with jpm_qubit_experiments.JPMQubitReadout() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    # run.sweep('Bias Voltage', np.linspace(-.45, -.3, 101) * V,
        # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])   
        
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.35, .55, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])

    # run.variable('RF Attenuation', 63);
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.46, 0.5, 201) * DACUnits,
           # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.sweep('RF Frequency', np.linspace(2, 6, 201) * GHz,
        # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])

    # run.sweep(['Fast Pulse Amplitude', 'RF Frequency'], [np.linspace(0.47, 0.485, 51)*DACUnits, np.linspace(3, 6., 151) * GHz],
       # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.sweep('Qubit Flux Bias Voltage', np.linspace(0, 1, 1001) * V,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])

    # run.sweep(['Input Bias Voltage', 'Bias Voltage'], 
              # [np.linspace(-1, 0, 201) * V, np.linspace(0, 1, 201) * V],
              # save=True, print_data=['Switching Probability'])
                  
    # run.sweep(['Input Bias Voltage', 'Bias Voltage'], 
              # [np.linspace(0, 1, 201) * V, np.linspace(0, -1, 201) * V],
              # save=True, print_data=['Switching Probability'])
              
    # run.sweep(['Input Bias Voltage', 'Bias Voltage'], 
          # [np.linspace(-1, 1, 201) * V, np.linspace(-1, 1, 201) * V],
          # save=True, print_data=['Switching Probability'])