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
                            'Leiden Board DAC 4',
                          ],
                'Leiden Board DAC 3':  {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Leiden Board DAC 4': {
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I',
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
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {
                                    'Readout Power': {'Setting': 'Power'}, 
                                    'Readout Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # GPIB RF Generator.
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
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
                    'Serial Number': 7033,
                    'Variables': ['Readout Attenuation']
                },
                { # Lab Brick Attenuator.
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
                                  'Actual Reps',
                                  'Threshold'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH048B-051215A-D10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Matched JPM Photon Counting\Leiden DR 2015-09-03 - Qubits and JPMs',
            'Experiment Name': 'ROFreqFluxBias2D',
            'Comments': 'MH048B Qubit and 051215A-D10 JPM in a single box. CW RF drive' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 4000, # should not exceed ~50,000
          
            'Qubit Frequency': 20 * GHz,
            'Qubit Power': -110 * dBm, 
            'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0.5 * DACUnits,
            'Qubit Time': 60 * ns,
            
            'Qubit Drive to Readout': 0 * ns,
            
            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 4.836 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 16 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 0 * MHz, 
            'Readout Amplitude': 0.5 * DACUnits,
            'Readout Time': 1000 * ns,
            'Readout Phase': 0 * rad,
            
            'Readout to Displacement': 0 * ns,
            'Readout to Displacement Offset': 0.00 * DACUnits,
            
            'Displacement Amplitude': 0.0 * DACUnits,
            'Displacement Time': 0 * ns,
            'Displacement Phase': 0 * rad,
            
            'Displacement to Fast Pulse': -200 * ns,  # time delay between the end of the displacement pulse and the start of the fast pulse
          
            'Init Time': 1000 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'Bias Voltage': 0.195 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': .1492 * DACUnits,
            'Fast Pulse Width': 0 * ns,
          
            'Threshold': 100 * PreAmpTimeCounts,
           }

with jpm_qubit_experiments.JPMQubitReadout() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    run.sweep('Bias Voltage', np.linspace(0.15, .25, 201) * V,
        save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])   
    
    # run.value('Readout Attenuation', 63*dB)
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.12, .16, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    #run.value('Readout Attenuation', 20*dB)
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.12, .16, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
          
    # run.value('Readout Attenuation', 15*dB)
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.12, .16, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
          
    # run.value('Readout Attenuation', 10*dB)
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.12, .16, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.value('Readout Attenuation', 20*dB)    
    # run.sweep('Readout Frequency', np.linspace(4.81, 4.85, 201) * GHz,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.value('Readout Attenuation', 18*dB)    
    # run.sweep('Readout Frequency', np.linspace(4.81, 4.85, 201) * GHz,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
  
    # run.sweep('Readout Attenuation', np.linspace(0, 63, 64) * dB,
          # save=False, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
            # [np.linspace(10, 35, 26) * dB, np.linspace(4.81, 4.85, 201) * GHz],
            # save=True, print_data=['Switching Probability'])
            
    # run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'], 
        # [np.linspace(-3, 3, 15) * V, np.linspace(4.81, 4.86, 101) * GHz],
        # save=True, print_data=['Switching Probability'])
    
    # run.sweep('Qubit Flux Bias Voltage', np.linspace(0, 1, 1001) * V,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])