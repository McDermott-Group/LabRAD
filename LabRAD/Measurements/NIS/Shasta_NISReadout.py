# Read out of an NIS junction connected to a resonator.

import os
import numpy as np

from labrad.units import us, ns, mV, V, GHz, MHz, rad, dB, dBm, DACUnits

import nis_experiments


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Shasta Board DAC 9', 
                            'Shasta Board DAC 10',
                            'Shasta Board ADC 11'
                          ],
                'Shasta Board DAC 9': {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                      },
                'Shasta Board DAC 10': {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                      },
                'Shasta Board ADC 11': {
                                        'RunMode': 'demodulate', #'average'
                                        'FilterType': 'square',
                                        'FilterStartAt': 0 * ns,
                                        'FilterWidth': 4000 * ns,
                                        'FilterLength': 4000 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -50 * MHz,
                                        'ADCDelay': 0 * ns,
                                        'Data': True
                                      },
                'Variables': {  # These are default values, you should
                                # normally overwrite these values in
                                # your particular experiment run.
                                'Init Time': {'Value': 100 * us},
                                'NIS Bias Voltage': {'Value': 0 * V},
                                'NIS Bias Time': {'Value': 10 * us},
                                'Bias to Readout Delay': {'Value': 100 *ns},
                                'ADC Wait Time': {'Value': 0 * ns},
                             }
                },

                # { # GPIB RF Generator, 'Address' field is required only
                  # # when more than one GPIB RF generator is present.
                    # 'Interface': 'RF Generator',
                    # 'Variables': {  
                                    # 'RF Power': {'Setting': 'Power'}, 
                                    # 'RF Frequency': {'Setting': 'Frequency'}
                                 # }
                # },
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
            'User': 'Chris',
            'Base Path': 'Z:\mcdermott-group\Data\NIS Junctions',
            'Experiment Name': 'Test',
            'Comments': 'Test' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 4000, # should not exceed ~5,000, use argument "runs" in sweep parameters instead 

            'Init Time': 100 * us,

            'RF Frequency': 4.9188 * GHz,
            'RF Power': 10 * dBm,
            
            'NIS Bias Voltage': 1.2 * V, # -2.5 to 2.5 V or 0 to 5 V
            'NIS Bias Time': 10 * us,
            
            'Bias to Readout Delay': 100 * us,
     
            'ADC Wait Time': 0 * ns,
           }


with nis_experiments.NISReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.value('NIS Bias Voltage', 1 * V)
    
    run.sweep('Bias to Readout Delay', np.linspace(0, 100, 101) * us,
              print_data=['I', 'Q'], plot_data=['I', 'Q'], max_data_dim=1,
              save=True, runs=3)
    
    run.sweep(['Bias to Readout Delay', 'RF Frequency'],
              [np.linspace(0, 100, 101) * ns, np.linspace(4.9, 5, 101) * GHz],
               print_data=['I', 'Q'], plot_data=['I', 'Q'], 
               save=True, runs=3) # runs does ex: 3X 3000 reps