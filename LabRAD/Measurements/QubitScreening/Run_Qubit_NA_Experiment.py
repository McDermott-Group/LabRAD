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
            'Device Name': 'MH048C',
            'User': 'Guilhem Ribeill',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-09-03 - Qubits and JPMs',
            'Experiment Name': 'FrequencyFluxBiasScan',
            'Comments': 'MH048A Qubit measured with network analyzer and SIM928 ' 
           }
 
# Experiment Variables
ExptVars = {
            'NA Center Frequency': 4.914 * GHz,
            'NA Frequency Span': 20 * MHz,
            
            'NA Source Power': -56 * dBm,
            
            'NA Frequency Points': 801,
            'NA Average Points': 250,
            
            'Qubit Flux Bias Voltage': 0 * V 
           }

with qubit_na_experiment.QubitNAExperiment() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    
    run.sweep('Qubit Flux Bias Voltage', np.linspace(-5, 5, 101) * V, save=True)

    #run.sweep('Qubit Flux Bias Voltage', np.array([0]) * V, save=True)