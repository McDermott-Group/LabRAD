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
                                        'DAC A': 'None',
                                        'DAC B': 'None',
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
                # { # GPIB RF Generator.
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    # 'Variables': {
                                    # 'RF Power': {'Setting': 'Power'}, 
                                    # 'RF Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
                # { # Lab Brick Attenuator.
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7032,
                    # 'Variables': ['RF Attenuation']
                # },
                # { # SIM Voltage Source.
                    # 'Interface': 'SIM928 Voltage Source',
                    # 'Address': ('SIM900 - ' + comp_name + 
                                # ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    # 'Variables': 'DC Bias Voltage'
                # },
                # { # SIM Voltage Source.
                    # 'Interface': 'SIM928 Voltage Source',
                    # 'Address': ('SIM900 - ' + comp_name + 
                                # ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    # 'Variables': 'JPM A Bias Voltage'
                # },
                # { # SIM Voltage Source.
                    # 'Interface': 'SIM928 Voltage Source',
                    # 'Address': ('SIM900 - ' + comp_name + 
                                # ' GPIB Bus - GPIB0::26::INSTR::SIM900::5'),
                    # 'Variables': 'JPM B Bias Voltage'
                # },
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
            'Experiment Name': 'FPDelay1D',
            'Comments': '051215A-D6 = JPM A, DAC 3 A+; 051215A-E10 = JPM B, DAC 3 B+. Both JPM A and JPM B inputs are open. FastBias cards in Fine mode. DC lines terminated with 50 Ohm. DB-25 E & D filters connected, both on top of the fridge. 50 Ohm terminations on unused RF lines. Foil. Oscilloscope disconnected.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 1000, # should not exceed ~50,000

            # 'RF Frequency': 20 * GHz,
            # 'RF Power': -110 * dBm,
            # 'RF Attenuation': 63 * dB, # should be in (0, 63] range
            # 'RF SB Frequency': 0 * MHz, 
            'RF Amplitude': 0 * DACUnits,
            'RF Time': 11000 * ns,
            
            # 'DC Bias Voltage': 0.1 * V,

            'Init Time': 500 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'JPM A Bias Voltage': .209 * V,
            'JPM A Fast Pulse Time': 10 * ns,
            'JPM A Fast Pulse Amplitude': .1085 * DACUnits,
            'JPM A Fast Pulse Width': 0 * ns,
            
            'JPM B Bias Voltage': .196 * V,
            'JPM B Fast Pulse Time': 10 * ns,
            'JPM B Fast Pulse Amplitude': .0829 * DACUnits,
            'JPM B Fast Pulse Width': 0 * ns,

            # Both JPM A and JPM B fast pulses should appear within
            # the RF Pulse, i.e. they should only start after the
            # beginning of the RF Pulse and finish before the end of
            # the RF Pulse.
            'RF to JPM A Fast Pulse Delay': 5500 * ns,
            'JPM A to JPM B Fast Pulse Delay': 0 * ns,
          
            'Threshold': 1253 * PreAmpTimeCounts,
           }

with double_jpm_experiments.DoubleJPMCorrelationFine() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    # run.sweep([['JPM A Bias Voltage'], ['JPM B Bias Voltage']],
            # [[np.linspace(.18, .25, 151) * V], [np.linspace(.18, .25, 151) * V]], 
            # save=True, print_data=['Pa', 'Pb', 'Temperature'], plot_data=['Pa', 'Pb'],
            # dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.sweep([['JPM A Fast Pulse Amplitude'], ['JPM B Fast Pulse Amplitude']],
            # [[np.linspace(.06, .12, 101) * DACUnits], [np.linspace(.06, .12, 101) * DACUnits]], 
            # save=True, print_data=['Pa', 'Pb', 'P11'], plot_data=['Pa', 'Pb', 'P11'],
            # dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.sweep(['JPM A Bias Voltage', 'JPM B Bias Voltage'], 
            # [np.linspace(0, .25, 251) * V, np.linspace(0, .25, 251) * V],
            # save=True, print_data=['Pa', 'Pb'])
    
    # run.sweep(['JPM A Fast Pulse Amplitude', 'JPM B Fast Pulse Amplitude'], 
            # [np.linspace(.096, .12, 51) * DACUnits, np.linspace(.068, .095, 51) * DACUnits],
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb'])
            
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 251) * ns, 
        # save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 126) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-5000, 5000, 501) * ns, 
            save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.value('Reps', 2500)
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 51) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('RF Frequency', np.linspace(3, 7, 401) * GHz, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('Reps', 30 * np.power(2, np.linspace(0, 10, 11)), 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])