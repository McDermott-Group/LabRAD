# read out of an NIS junction connected to a resonator.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import hemt_qubit_experiments
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
                'Variables': {
                                'Init Time': {'Value': 100 * us},
                                'NIS Bias Voltage': {'Value': 0 * V},
                                'NIS Bias Time': {'Value': 10 * us},
                                'Bias to Readout Delay': {'Value': 100 *ns},
                                'ADC Wait Time': {'Value': 0 * ns},
                                'ADC Demod Frequency': {'Value': -50 * MHz}
                             }
                },

                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::19::INSTR',
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
                { # ADR3
                    'Interface': 'ADR3',
                    'Variables': {'Temperature': {'Setting':'Temperature', 'Stage': 'FAA'}}
                },
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
            'Experiment Name': 'CavityEvolution',
            'Comments': 'ADC band pass filters removed.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 3000, # should not exceed ~5,000, use agrument "runs" in sweep parameters instead 

            'Init Time': 100 * us,
            
            # 'Stark Amplitude': 0 * DACUnits,
            # 'Stark Time': 10000 * ns,

            'RF Frequency': 4.9188 * GHz, #4.9139 * GHz,
            'RF Power': 10 * dBm,
            
            'NIS Bias Voltage': 1.2 * V, # -2.5 to 2.5V
            'NIS Bias Time': 10 * us,
            
            'Bias to Readout Delay': 100 * ns,
     
            'ADC Wait Time': 0 * ns, # time delay between the start of the readout pulse and the start of the demodulation
            # 'ADC Demod Frequency': -50 * MHz
           }


with nis_experiments.NISReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.value('NIS Bias Voltage', 1 * V)
    
    run.sweep('Bias to Readout Delay', np.linspace(0, 100, 101)*ns, # for >1D, do ['list','of','variables'],[linespaces]
                print_data=['I', 'Q'], plot_data=['I', 'Q'], save=True, runs=3) # runs does ex: 3X 3000 reps
    
    