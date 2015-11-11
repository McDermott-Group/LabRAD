# Read two JPMs in a correlation experiment.

import os
import numpy as np

from labrad.units import (us, ns, V, mV, GHz, MHz, rad, dB, dBm,
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
                                'RF Time': {'Value': 11000 * ns},
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
                                  'Min Threshold',
                                  'Max Threshold',
                                  'Preamp Timeout'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'NIST040115-1 = 051215A-E11+E10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Matched JPM Photon Counting\Leiden DR 2015-10-02 - Cross-Correlation',
            'Experiment Name': 'FPDelay1D',
            'Comments': '051215A-E11 = JPM A, DAC 3 A+; 051215A-E10 = JPM B, DAC 3 B+. Both JPM A and JPM B inputs are connected to a tunnel junction via a bias-T, a circulator and a splitter. FastBias cards in Fine mode. DC lines terminated with 50/0 Ohm. DB-25 E & D filters on top of the fridge. 50 Ohm terminations on unused RF lines. Oscilloscope connected.' 
           }
# PA2: 10, 22, 33700, FB15, 33, 22, 33650
# PA5: 10, 22, 33890, FB17, 33, 22, 34230

# Experiment Variables
ExptVars = {
            'Reps': 25000, # should not exceed ~50,000

            'RF Frequency': 20 * GHz, #3.54 * GHz,
            'RF Power': -110 * dBm,
            'RF Attenuation': 63 * dB, #40 * dB, # should be in (0, 63] range
            # 'RF SB Frequency': 0 * MHz, 
            # 'RF Amplitude': 0.5 * DACUnits,
            # 'RF Time': 11000 * ns,
            
            'DC Bias Voltage': 0 * mV,

            'Init Time': 500 * us,
            'Bias Time': 150 * us,
            'Measure Time': 20 * us,
          
            'JPM A Bias Voltage': .2842 * V,
            'JPM A Fast Pulse Time': 3 * ns,
            'JPM A Fast Pulse Amplitude': 1 * DACUnits,
            'JPM A Fast Pulse Width': 0 * ns,
            
            'JPM B Bias Voltage': .1565 * V,
            'JPM B Fast Pulse Time': 3 * ns,
            'JPM B Fast Pulse Amplitude': 1 * DACUnits,
            'JPM B Fast Pulse Width': 0 * ns,

            # Both JPM A and JPM B fast pulses should appear within
            # the RF Pulse, i.e. they should only start after the
            # beginning of the RF Pulse and finish before the end of
            # the RF Pulse.
            'RF to JPM A Fast Pulse Delay': 5500 * ns,
            'JPM A to JPM B Fast Pulse Delay': 0 * ns,

            'Min Threshold': 0 * PreAmpTimeCounts,
            'Max Threshold': 503 * PreAmpTimeCounts,
           }

with double_jpm_experiments.DoubleJPMCorrelation() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars) 

    # run.sweep('JPM B Bias Voltage', np.linspace(.17, .18, 251) * V,
            # save=True, print_data=['Pa', 'Pb', 'P11'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep([['JPM A Bias Voltage'], ['JPM B Bias Voltage']],
            # [[np.linspace(.275, .32, 51) * V], [np.linspace(0*.155, 0*.178, 51) * V]], 
            # save=True, print_data=['Pa', 'Pb', 'Temperature'], plot_data=['Pa', 'Pb'],
            # dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.sweep([['JPM A Fast Pulse Amplitude'], ['JPM B Fast Pulse Amplitude']],
            # [[np.linspace(0.7, 1, 51) * DACUnits], [np.linspace(0.7, 1, 51) * DACUnits]], 
            # save=True, print_data=['Pa', 'Pb', 'P11'], plot_data=['Pa', 'Pb', 'P11'],
            # dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.sweep(['JPM A Bias Voltage', 'JPM B Bias Voltage'], 
            # [np.linspace(0.29, .32, 151) * V, np.linspace(0.175, .177, 51) * V],
            # save=True, print_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep(['JPM A Fast Pulse Amplitude', 'JPM B Fast Pulse Amplitude'], 
            # [np.linspace(.096, .12, 51) * DACUnits, np.linspace(.068, .095, 51) * DACUnits],
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb'])
            
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 251) * ns, 
        # save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-10, 10, 21) * ns, 
        # save=True, print_data=['Pa', 'Pb', 'P11', 'Corr Coef', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-50, 50, 51) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-5000, 5000, 1001) * ns, 
            save=True, print_data=['Pa', 'Pb', 'P11', 'Temperature'], plot_data=['Pa', 'Pb', 'P11'])
    
    # run.value('Reps', 2500)
    # run.sweep('JPM A to JPM B Fast Pulse Delay', np.linspace(-500, 500, 51) * ns, 
            # save=True, print_data=['Pa', 'Pb', 'P11'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('RF Attenuation', np.linspace(1, 63, 63) * dB, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11', 'Corr Coef'])
    
    # run.value('RF Attenuation', 15 * dB)
    # run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.value('RF Attenuation', 20 * dB)
    # run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.value('RF Attenuation', 25 * dB)
    # run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.value('RF Attenuation', 30 * dB)
    # run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])

    # run.value('RF Attenuation', 35 * dB)
    # run.sweep('RF Frequency', np.linspace(3, 7, 201) * GHz, 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('Reps', 30 * np.power(2, np.linspace(0, 10, 11)), 
            # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11'])
            
    # run.sweep('DC Bias Voltage', np.linspace(-30, 30, 61) * mV, 
        # save=True, print_data=['Pa', 'Pb'], plot_data=['Pa', 'Pb', 'P11', 'Temperature'])

    # run.sweep(['RF Attenuation', 'RF Frequency'],
              # [np.linspace(30, 63, 6) * dB, np.linspace(3, 7, 201) * GHz], 
              # save=True, print_data=['Pa', 'Pb', 'P11'])
    
    # run.sweep(['RF Attenuation', 'DC Bias Voltage'], 
              # [np.linspace(15, 35, 11) * dB, np.linspace(0, 30, 31) * mV], 
              # save=True, print_data=['Pa', 'Pb', 'P11'])
   
    # run.value('RF Attenuation', 25 * dB)   
    # run.sweep(['DC Bias Voltage', 'RF Frequency'],
              # [np.linspace(0, 30, 16) * mV, np.linspace(3, 7, 201) * GHz], 
              # save=True, print_data=['Pa', 'Pb', 'P11'])