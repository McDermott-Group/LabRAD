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
                            'Leiden Board DAC 3',
                            'Leiden Board DAC 4'
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
                                'Readout to Fast Pulse': {'Value': 0 * ns},
                                'Stark Amplitude': {'Value': 0 * DACUnits},
                                'Stark Time': {'Value': 0 * ns}
                             }
                },
                # { # GPIB RF Generator.
                    # 'Interface': 'RF Generator',
                    # 'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {
                                    # 'Qubit Power': {'Setting': 'Power'}, 
                                    # 'Qubit Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {  
                                    'Qubit Power': {'Setting': 'Power'}, 
                                    'Qubit Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab RF Generator
                    'Interface': 'Lab Brick RF Generator',
                    'Serial Number': 10776,
                    'Variables': {  
                                    'Readout Power': {'Setting': 'Power'}, 
                                    'Readout Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': 'Readout Attenuation'
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7031,
                    'Variables': 'Qubit Attenuation'
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
                    'Variables': {
                                    'Reps': {},
                                    'Actual Reps': {},
                                    'Threshold': {},
                                    'Preamp Timeout': {},
                                    'Histogram': {'Value': False}
                                  }
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH060-051215A-E11',
            'User': 'Guilhem Ribeill',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'StarkShift',
            'Comments': 'Hittite amps removed from setup, replaced the readout IQ-mixer, "swap" qubit and readout drives, trying fast readout.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 2500, # should not exceed ~55,000

            # 'RF Frequency': 20 * GHz,
            # 'RF Power': -40 * dBm,
            # 'RF Attenuation': 63 * dB, # should be in (0, 6s3] range
          
            'Stark Amplitude': 1 * DACUnits,
            'Stark Time': 10 * us,
            'Readout to Fast Pulse': 0 * ns,
          
            'Qubit Frequency': 4.448 * GHz,
            'Qubit Power': 13 * dBm,
            'Qubit Attenuation': 9 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 1.0 * DACUnits,
            'Qubit Time': 4000 * ns,
            
            'Qubit Drive to Readout': 0 * ns,
            
            'Qubit Flux Bias Voltage': 0.4 * V,

            'Readout Frequency': 4.9092 * GHz,
            'Readout Power': 10 * dBm,
            'Readout Attenuation': 17 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 62.5 * MHz, 
            'Readout Amplitude': 1.0 * DACUnits,
            'Readout Time': 150 * ns,
            'Readout Phase': 0 * rad,
            
            'Readout to Displacement': 0 * ns,
            # 'Readout to Displacement Offset': 0.0 * DACUnits,
            
            # 'Displacement Amplitude': 0.0 * DACUnits,
            # 'Displacement Time': 0 * ns,
            # 'Displacement Phase': 0 * rad,
            
            'Displacement to Fast Pulse': -100 * ns,  # time delay between the end of the displacement pulse and the start of the fast pulse
          
            'Init Time': 2000 * us,
            'Bias Time': 100 * us,
            'Measure Time': 8 * us,
          
            'Bias Voltage': 0.182 * V,
            'Fast Pulse Time': 150 * ns,
            'Fast Pulse Amplitude': .5335 * DACUnits,
            'Fast Pulse Width': 0 * ns,
           }

#with jpm_qubit_experiments.JPMQubitReadout() as run:
with jpm_qubit_experiments.JPMStarkShift() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.sweep(['Qubit Frequency', 'Stark Amplitude'],
            [np.linspace(4.25, 4.55, 151)*GHz, np.linspace(0,0.5,26)*DACUnits],
            save=True, print_data=['Switching Probability'])
    
    # run.sweep('Bias Voltage', np.linspace(0.14, .24, 51) * V,
        # save=False, print_data=['Switching Probability'], plot_data=['Switching Probability'])   
    
    # run.sweep('Fast Pulse Amplitude', np.linspace(0.49, .58, 51) * DACUnits,
          # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])

    # run.sweep('Init Time', np.array([20, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000]) * us,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])
    
    # run.sweep('RF Frequency', np.linspace(3, 7, 401) * GHz,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    
    # run.sweep('Readout Frequency', np.linspace(4.905, 4.925, 201) * GHz,
        # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    # run.value('Qubit Time', 0 * ns)      
    # run.sweep('Readout Frequency', np.linspace(4.905, 4.925, 201) * GHz,
        # save=True, print_data='Switching Probability', plot_data='Switching Probability')

    # run.sweep('Readout Time', np.linspace(10, 2000, 100) * ns,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
            
    # run.sweep('Readout Attenuation', np.linspace(1, 50, 50) * dB,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')    
    
    # run.value('Qubit Frequency', 4.65*GHz) 
    # run.sweep('Readout Frequency', np.linspace(4.91, 4.92, 51) * GHz,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
            
    # run.sweep('Qubit Frequency', np.linspace(4.4, 4.5, 201) * GHz,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    
    # run.sweep('Qubit Time', np.linspace(0,1000,501) * ns,
                # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    
    # run.sweep(['Qubit Attenuation', 'Qubit Frequency'], 
            # [np.linspace(1, 21, 11) * dB, np.linspace(4.1, 4.35, 351) * GHz],
            # save=True, print_data=['Switching Probability'])
            
    # run.value('Qubit Frequency', 4.177 * GHz)              
    # run.sweep(['Qubit Frequency', 'Qubit Time'],[np.linspace(4.1, 4.3, 101)*GHz, np.linspace(0, 500, 251) * ns],
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    
    # run.sweep('Displacement to Fast Pulse', np.linspace(-200, 200, 101) * ns,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')          

    # run.sweep('Readout Time', np.linspace(0, 1200, 101) * ns,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')   
    
    # run.sweep('Qubit Drive to Readout', np.linspace(0, 1000, 501) * ns,
            # save=True, print_data='Switching Probability', plot_data='Switching Probability')
    
    # run.sweep(['Qubit Attenuation', 'Qubit Time'], 
            # [np.linspace(1, 9, 5) * dB, np.linspace(0, 150, 76) * ns],
            # save=True, print_data=['Switching Probability'])
 
    # run.sweep('Readout Attenuation', np.linspace(1, 61, 15) * dB,
        # save=False, print_data='Switching Probability', plot_data='Switching Probability')

    # run.sweep('Qubit Attenuation', np.linspace(1, 40, 40) * dB,
        # save=True, print_data='Switching Probability', plot_data='Switching Probability')
        
    # run.sweep('Readout Power', np.linspace(13, -47, 15) * dBm,
        # save=False, print_data='Switching Probability', plot_data='Switching Probability')
    
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
            # [np.linspace(10, 30, 11) * dB, np.linspace(4.90, 4.92, 101) * GHz],
            # save=True, print_data=['Switching Probability'])

    # run.value('Qubit Time', 0*ns)
    
    # run.sweep(['Readout Attenuation', 'Readout Frequency'], 
            # [np.linspace(10, 30, 11) * dB, np.linspace(4.90, 4.92, 101) * GHz],
            # save=True, print_data=['Switching Probability'])
    
    # run.sweep(['Readout Power', 'Readout Frequency'], 
            # [np.linspace(13, -42, 12) * dBm, np.linspace(4.35, 4.45, 51) * GHz],
            # save=True, print_data=['Switching Probability'])
            
    # run.sweep([['Readout Attenuation', 'Readout Frequency'], ['Qubit Attenuation', 'Qubit Frequency']], 
        # [[np.linspace(10, 40, 11) * dB, np.linspace(4.4, 5.1, 351) * GHz], [np.linspace(10, 40, 11) * dB, np.linspace(4.4, 5.1, 351) * GHz]],
        # save=True, print_data=['Switching Probability'])
            
    # run.sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'], 
        # [np.linspace(-2, 2, 51) * V, np.linspace(4.912, 4.92, 81) * GHz],
        # save=True, print_data=['Switching Probability'])
        
    # run.sweep(['Qubit Flux Bias Voltage', 'Qubit Frequency'], 
        # [np.linspace(0, 1., 6) * V, np.linspace(3.8, 4.6, 401) * GHz],
        # save=True, print_data=['Switching Probability'])
    
    # run.sweep('Qubit Flux Bias Voltage', np.linspace(0, 1, 1001) * V,
            # save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])