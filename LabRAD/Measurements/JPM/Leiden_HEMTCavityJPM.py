# Probe a resonator driven by switching JPM with a HEMT.

import os
import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import hemt_qubit_experiments

                          
comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Leiden Board DAC 3', 
                            'Leiden Board DAC 4',
                            'Leiden Board ADC 5'
                          ],
                'Leiden Board DAC 3':  {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                       },
                'Leiden Board DAC 4': {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                       },
                'Leiden Board ADC 5': {
                                        'RunMode': 'demodulate', #'average'
                                        'FilterType': 'square',
                                        'FilterWidth': 9500 * ns,
                                        'FilterLength': 10000 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': 0 * MHz,
                                        'ADCDelay': 0 * ns,
                                        'Data': True
                                       },
                'Variables': {
                                'Init Time': {},
                                'Bias Time': {},
                                'Measure Time': {},
                                'Bias Voltage': {},
                                'Fast Pulse Time': {},
                                'Fast Pulse Amplitude': {},
                                'Fast Pulse Width': {'Value': 0 * ns},
                                'ADC Wait Time': {'Value': 0 * ns}
                             }
                },
                { # GPIB RF Generator
                    'Interface': 'RF Generator',
                    'Address': comp_name + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {  
                                    'RF Power': {'Setting': 'Power'}, 
                                    'RF Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': ['RF Attenuation']
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
                                  'Rep Iteration'
                                  'Runs',
                                  'ADC Time'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH048B-051215A-D10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Leiden DR 2015-09-25 - Qubits and JPMs',
            'Experiment Name': 'InverseDirectionCavitySpectroscopy',
            'Comments': 'Driving qubit cavity through JPM Fast Pulse line.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 5000, # should not exceed ~50,000

            'Qubit Flux Bias Voltage': 0 * V,

            'RF Frequency': 4.821 * GHz,
            'RF Power': 13 * dBm,

            'Init Time': 100 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'Bias Voltage': 0.195 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': .1492 * DACUnits,
            'Fast Pulse Width': 0 * ns,

            'ADC Wait Time': 1000 * ns, # time delay between the start of the readout pulse and the start of the demodulation
           }

with hemt_qubit_experiments.HEMTCavityJPM() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)

    run.run_once(plot_waveforms=True)
    
    run.single_shot_iqs(save=False, plot_data=True)
    # run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=True, plot_data=['I', 'Q'], runs=250)

    # run.sweep('Readout Frequency', np.linspace(4.65, 4.85, 201) * GHz,
              # plot_data=['I', 'Q', 'ADC Amplitude'], save=True)