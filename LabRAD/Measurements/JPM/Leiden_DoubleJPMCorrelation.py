# Read two JPMs in a correlation experiment.

import os
import numpy as np

from labrad.units import (us, ns, V, mV, GHz, MHz, rad, mK, dB, dBm,
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
            'Device Name': 'NIST040115-1 = 051215A-E11+E10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Matched JPM Photon Counting\Leiden DR 2015-10-02 - Cross-Correlation',
            'Experiment Name': 'RFFreq1D',
            'Comments': '051215A-E11 = JPM A, DAC 3 A+; 051215A-E10 = JPM B, DAC 3 B+. Both JPM A and JPM B inputs are connected to a tunnel junction via a bias-T, a circulator and a splitter. FastBias cards in Fine mode. DC lines terminated with 50/0 Ohm. DB-25 E & D filters on top of the fridge. 50 Ohm terminations on unused RF lines. Oscilloscope connected.' 
           }
# PA2: 10, 22, 33700, FB15
# PA5: 10, 22, 33890, FB17 

           
# Experiment Variables
ExptVars = {
            'Reps': 30000, # should not exceed ~50,000

            'RF Frequency': 5 * GHz,
            'RF Power': 13 * dBm,
            'RF Attenuation': 63 * dB, # should be in (0, 63] range
            'RF SB Frequency': 0 * MHz, 
            'RF Amplitude': 0.5 * DACUnits,
            'RF Time': 11000 * ns,
            
            'DC Bias Voltage': 0 * mV,

            'Init Time': 500 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'JPM A Bias Voltage': .179 * V,
            'JPM A Fast Pulse Time': 10 * ns,
            'JPM A Fast Pulse Amplitude':  .7761 * DACUnits,
            'JPM A Fast Pulse Width': 0 * ns,
            
            'JPM B Bias Voltage': .195 * V,
            'JPM B Fast Pulse Time': 10 * ns,
            'JPM B Fast Pulse Amplitude': .7333 * DACUnits,
            'JPM B Fast Pulse Width': 0 * ns,

            # Both JPM A and JPM B fast pulses should appear within
            # the RF Pulse, i.e. they should only start after the
            # beginning of the RF Pulse and finish before the end of
            # the RF Pulse.
            'RF to JPM A Fast Pulse Delay': 5500 * ns,
            'JPM A to JPM B Fast Pulse Delay': 0 * ns,
          
            'Threshold': 100 * PreAmpTimeCounts,
           }

with double_jpm_experiments.DoubleJPMCorrelation() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    # run.sweep([['JPM A Bias Voltage'], ['JPM B Bias Voltage']],
            # [[np.linspace(0.16, 0.22, 151) * V], [np.linspace(0.16, 0.22, 151) * V]], 
            # save=True, print_data=['Pa', 'Pb', 'Temperature'], plot_data=['Pa', 'Pb'],
            # dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.sweep([['JPM A Fast Pulse Amplitude'], ['JPM B Fast Pulse Amplitude']],
            # [[np.linspace(.72, .79, 101) * DACUnits], [np.linspace(.72, .79, 101) * DACUnits]], 
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
    
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-10, 10, 21) * ns, 
        # save=True, print_data=['Pa', 'Pb', 'P11', 'Corr Coef', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 126) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-5000, 5000, 501) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.value('Reps', 2500)
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 51) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('RF Attenuation', np.linspace(1, 63, 63) * dB, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11', 'Corr Coef'])
    
    run.value('RF Attenuation', 15 * dB)
    run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    run.value('RF Attenuation', 20 * dB)
    run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    run.value('RF Attenuation', 25 * dB)
    run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    run.value('RF Attenuation', 30 * dB)
    run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])

    run.value('RF Attenuation', 35 * dB)
    run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('Reps', 30 * np.power(2, np.linspace(0, 10, 11)), 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])