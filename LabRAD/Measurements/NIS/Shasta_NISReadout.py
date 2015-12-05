# Read out of an NIS junction connected to a resonator.

import os
import numpy as np

from labrad.units import us, ns, mV, V, GHz, MHz, rad, dB, dBm, DACUnits

import nis_experiments


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'mcdermott5125 Board DAC 9', 
                            'mcdermott5125 Board DAC 10',
                            'mcdermott5125 Board ADC 11'
                          ],
                'mcdermott5125 Board DAC 9': {
                                        'DAC A': 'RF I',    # DAC A: 0-1V (???)
                                        'DAC B': 'RF Q',    # DAC B: -2.5 to 2.5 V or 0 to 5 V (???)
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                      },
                'mcdermott5125 Board DAC 10': {
                                        'DAC A': 'RF I',
                                        'DAC B': 'RF Q',
                                      },
                'mcdermott5125 Board ADC 11': {
                                        'RunMode': 'demodulate', #'average'
                                        'FilterType': 'square',
                                        'FilterStartAt': 4000 * ns,
                                        'FilterWidth': 10000 * ns, # ignore if 'FilterType' is 'square'.
                                        'FilterLength': 5960 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -30 * MHz,
                                        'ADCDelay': 0 * ns,
                                        'Data': True
                                      },
                'Variables': {  # These are default values, you should
                                # normally overwrite these values in
                                # your particular experiment run.
                                'Init Time': {'Value': 100 * us},
                                'NIS Bias Voltage': {'Value': 0 * V},
                                'NIS Bias Time': {'Value': 10 * us},
                                'RF Amplitude': {'Value': 0 * DACUnits},
                                'RF Time': {'Value': 0 * ns},
                                'RF SB Frequency': {'Value': 32.5 * MHz},
                                'Bias to RF Delay': {'Value': 0 * us},
                                'ADC Wait Time': {'Value': 0 * ns},
                             }
                },
                { # GPIB RF Generator, 'Address' field is required only
                  # when more than one GPIB RF generator is present.
                    'Interface': 'RF Generator',
                    'Variables': {  
                                    'RF Power': {'Setting': 'Power'}, 
                                    'RF Frequency': {'Setting': 'Frequency'}
                                 }
                },
                # { # Lab Brick Attenuator
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7031,
                    # 'Variables': 'Qubit Attenuation'
                # },
                # { # SIM Voltage Source.
                    # 'Interface': 'SIM928 Voltage Source',
                    # 'Address': ('SIM900 - ' + comp_name + 
                                # ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    # 'Variables': 'Qubit Flux Bias Voltage'
                # },
                # { # ADR3
                    # 'Interface': 'ADR3',
                    # 'Variables': {'Temperature': {'Setting':'Temperatures', 'Stage': 'FAA'}}
                # },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Reps', 'Runs'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'NIS1',
            'User': 'Umesh',
            'Base Path': 'Z:\mcdermott-group\Data\NIS Junctions',
            'Experiment Name': 'RFFreq2DSweepFine',
            'Comments': '2D sweep BiastoRFdelay Frequency' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 50, # should not exceed ~5,000, use argument "runs" in sweep parameters instead 

            'Init Time': 3000 * us,

            'RF Frequency': 4.6736 * GHz,
            'RF Power': 17 * dBm, #17.6 * dBm,
            'RF Time': 10000 * ns,
            'RF SB Frequency': 30 * MHz,
            'RF Amplitude': 0.5 * DACUnits, # [-1, 1] * DACUnits, 1 DACUnits ~ 0.1-2.0 V
            
            'NIS Bias Voltage': 0.3 * V, # -2.5 to 2.5 V or 0 to 5 V
            'NIS Bias Time': 300 * us,
            
            'Bias to RF Delay': 100 * us,
     
            'ADC Wait Time': -100 * ns,
           }


with nis_experiments.NISReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    #run.single_shot_iqs(save=False, plot_data=True)
    #run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=True, plot_data=['I', 'Q'], runs=1000)
    
    # run.value('NIS Bias Voltage', 0.0 * V)
   
    
    # run.sweep('RF Amplitude', np.linspace(0, 1, 31) * DACUnits,
              # print_data=['I', 'Q'], plot_data=['I', 'Q'], max_data_dim=1,
              # save=False, runs=1)
    
    # run.sweep('RF Frequency', np.linspace(4.673, 4.6744, 101) * GHz,
              # plot_data=['I', 'Q', 'Amplitude'], max_data_dim=1,
              # save=True, runs=1)
    
    # run.sweep(['Bias to Readout Delay', 'RF Frequency'],s
              # [np.linspace(0, 100, 101) * us, np.linspace(4.9, 5, 101) * GHz],
               # print_data=['I', 'Q'], plot_data=['I', 'Q'], 
               # save=True, runs=3) # runs does ex: 3X 4000 reps
               
   
    #run.sweep(['Bias to RF Delay', 'RF Frequency'], 
         # [np.linspace(0, 100, 11) * us, np.linspace(4.6725, 4.6744, 200) * GHz],
         # save=True, runs=1)
   
    bias_range = np.linspace(0, 0.5, 11) * V
    for voltage in bias_range:
        run.value('NIS Bias Voltage', voltage)
        run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  [np.linspace(0, 50, 21) * us, np.linspace(4.669, 4.675, 150) * GHz],
                  save=True, runs=1)
                  
    # bias_range = np.linspace(0, 1, 11) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)
        # run.sweep('RF Frequency', np.linspace(4.673, 4.6744, 101) * GHz,
          # plot_data=['I', 'Q', 'Amplitude'], max_data_dim=1,
          # save=True, runs=1)