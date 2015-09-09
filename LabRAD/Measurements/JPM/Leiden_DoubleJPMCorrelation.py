# Read two JPMs in a correlation experiment.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import double_jpm_experiments


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Leiden Board DAC 3', 
                            'Leiden Board DAC 4',
                          ],
                'Leiden Board DAC 3':  {
                                        'DAC A': 'JPM A Fast Pulse',
                                        'DAC B': 'JPM B Fast Pulse',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Leiden Board DAC 4': {
                                        'DAC A': 'RF Q',
                                        'DAC B': 'RF I',
                                        'Data': True
                                       },
                'Variables': {  # Default values.
                                'Init Time': {},
                                'Bias Time': {},
                                'Measure Time': {},
                                'JPM A Bias Voltage': {'Value': 0 * V},
                                'JPM A Fast Pulse Time': {'Value': 0 * ns},
                                'JPM A Fast Pulse Amplitude': {'Value': 0 * DACUnits},
                                'JPM A Fast Pulse Width': {'Value': 0 * ns},
                                'JPM B Bias Voltage': {'Value': 0 * V},
                                'JPM B Fast Pulse Time': {'Value': 0 * ns},
                                'JPM B Fast Pulse Amplitude': {'Value': 0 * DACUnits},
                                'JPM B Fast Pulse Width': {'Value': 0 * ns},
                                'RF SB Frequency': {'Value': 0 * MHz},
                                'RF Amplitude': {'Value': 0 * DACUnits},
                                'RF Time': {'Value': 0 * ns},
                                'RF to JPM A Fast Pulse Delay': {},
                                'JPM A to JPM B Fast Pulse Delay': {'Value': 0 * ns}
                             }
                },
                { # GPIB RF Generator.
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {
                                    'RF Power': {'Setting': 'Power'}, 
                                    'RF Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator.
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': ['RF Attenuation']
                },
                { # SIM Voltage Source.
                    'Interface': 'SIM928 Voltage Source',
                    'Address': ('SIM900 - ' + comp_name + 
                                ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    'Variables': 'DC Bias Voltage'
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
            'Device Name': '051215A-D6 and 051215A-E10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Matched JPM Photon Counting\Leiden DR 2015-09-03 - Qubits and JPMs',
            'Experiment Name': 'BV1D',
            'Comments': '051215A-D6 = JPM A, 051215A-E10 = JPM B. JPM A input is connected to the calibration line via the codl relays. JPM B input is open.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 120, # should not exceed ~50,000

            'RF Frequency': 4.836 * GHz,
            'RF Power': 13 * dBm,
            'RF Attenuation': 16 * dB, # should be in (0, 63] range
            'RF SB Frequency': 0 * MHz, 
            'RF Amplitude': 0.5 * DACUnits,
            'RF Time': 1000 * ns,
            
            'DC Bias Voltage': 0 * V,

            'Init Time': 1000 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'JPM A Bias Voltage': 0 * V,
            'JPM A Fast Pulse Time': 10 * ns,
            'JPM A Fast Pulse Amplitude': .5 * DACUnits,
            'JPM A Fast Pulse Width': 0 * ns,
            
            'JPM B Bias Voltage': 0 * V,
            'JPM B Fast Pulse Time': 10 * ns,
            'JPM B Fast Pulse Amplitude': .5 * DACUnits,
            'JPM B Fast Pulse Width': 0 * ns,

            # Both JPM A and JPM B fast pulses should appear within
            # the RF Pulse, i.e. they should only start after the
            # beginning of the RF Pulse and finish before the end of
            # the RF Pulse.            
            'RF to JPM A Fast Pulse Delay': 1000 * ns,
            'JPM A to JPM B Fast Pulse Delay': 0 * ns,
          
            'Threshold': 100 * PreAmpTimeCounts,
           }

with double_jpm_experiments.DoubleJPMCorrelation() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    run.sweep([['JPM A Bias Voltage'], ['JPM B Bias Voltage']],
             [[np.linspace(0, .3, 151) * V], [np.linspace(0, .3, 151) * V]], 
             save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb'],
             dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.sweep('JPM A Bias Voltage', np.linspace(0.15, .25, 201) * V,
        # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])   
    
    # run.sweep('JPM A Fast Pulse Amplitude', np.linspace(0.12, .16, 101) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.sweep(['JPM A Bias Voltage', 'JPM B Bias Voltage'], 
        # [np.linspace(0, .5, 201) * V, np.linspace(0, .5, 201) * V],
        # save=True, print_data=['Switching Probability'])