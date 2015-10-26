# Measure SLUG microwave performance
import os
import numpy as np

from labrad.units import (us, ns, mV, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import slug_rf_experiment


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Shasta Board DAC 9',
                          ],
                'Shasta Board DAC 9':  {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Variables': {  # Default values.
                                'Init Time': {},
                                'Bias Time': {},
                                'Bias Voltage': {}
                             }
                },
                { # ADR3
                    'Interface': 'ADR3',
                    'Variables': {
                                    'Temperature': {'Setting': 'Temperatures',
                                                    'Stage': '3K'}
                                 }
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['NA Center Frequency',
                                    'NA Frequency Span',      
                                    'NA Source Power',
                                    'NA Frequency Points',
                                    'NA Average Points']
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': [
                                  'Reps',
                                  'Actual Reps',
                                 ],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'Test',
            'User': 'Test User',
            'Base Path': 'Z:\mcdermott-group\Data\Test',
            'Experiment Name': 'Joe is a cool guy',
            'Comments': 'What am I doing?' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 3010, # should not exceed ~50,000

            'Init Time': 500 * us,
            'Bias Time': 100 * us,
          
            'Bias Voltage': 500 * mV,
           }

with fim_experiment.FIM() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.sweep('Bias Time', np.linspace(100, 1100, 11) * us,
        save=True, print_data=['Temperature'], plot_data=['Temperature'])   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # Qubit spectroscopy with a network analyzer.

import os
import numpy as np

from labrad.units import V, GHz, MHz, mK, dB, dBm

import qubit_na_experiment


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   
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
                    'Variables': ['NA Center Frequency',
                                    'NA Frequency Span',      
                                    'NA Source Power',
                                    'NA Frequency Points',
                                    'NA Average Points']
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH048A',
            'User': 'Guilhem Ribeill',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-09-03 - Qubits and JPMs',
            'Experiment Name': 'FrequencyFluxBiasScan',
            'Comments': 'MH048A Qubit measured with network analyzer and SIM928 ' 
           }
 
# Experiment Variables
ExptVars = {
            'NA Center Frequency': 4.881 * GHz,
            'NA Frequency Span': 20 * MHz,
            
            'NA Source Power': -53 * dBm,
            
            'NA Frequency Points': 801,
            'NA Average Points': 250,
            
            'Qubit Flux Bias Voltage': 0 * V 
           }

with qubit_na_experiment.QubitNAExperiment() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    
    run.sweep('Qubit Flux Bias Voltage', np.linspace(-5, 5, 101) * V, save=True)

    #run.sweep('Qubit Flux Bias Voltage', np.array([0]) * V, save=True)
