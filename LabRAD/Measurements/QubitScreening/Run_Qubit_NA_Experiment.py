# Qubit spectroscopy with a network analyzer.

import os
import numpy as np

from labrad.units import V, GHz, MHz, dB, dBm

import qubit_na_experiment


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   
                { # SIM Voltage Source
                    'Interface': 'SIM928 Voltage Source',
                    'Address': ('SIM900 - ' + comp_name + 
                                ' GPIB Bus - GPIB0::26::INSTR::SIM900::3'),
                    'Variables': 'Qubit Flux Bias Voltage'
                },
                { # Network Analyzer
                    'Interface': 'Network Analyzer',
                    'Variables': {'NA Center Frequency': {'Setting': 'Center Frequency'},
                                  'NA Frequency Span': {'Setting': 'Span Frequency'},
                                  'NA Source Power': {'Setting': 'Source Power'},
                                  'NA Sweep Points': {'Setting': 'Sweep Points'},
                                  'NA Average Points': {'Setting': 'Average Points'},
                                  'NA Start Frequency': {'Setting': 'Start Frequency'},
                                  'NA Stop Frequency': {'Setting': 'Stop Frequency'},
                                  'Trace': {'Setting': 'Get Trace'}}
                },
                { # Leiden Fridge
                    'Interface': 'Leiden',
                    'Variables': {'Temperature': {'Setting': 'Mix Temperature'}}
                },
                { # Readings entered manually, software parameters
                    'Interface': None,
                    'Variables': []
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH061A',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'FrequencyFluxBias2D',
            'Comments': 'MH061A Qubit measured with network analyzer and SIM928.' 
           }
 
# Experiment Variables
ExptVars = {
            'NA Center Frequency': 4.943 * GHz,
            'NA Frequency Span': 60 * MHz,
            
            'NA Source Power': -58 * dBm,
            
            'NA Sweep Points': 1801,
            'NA Average Points': 1000,
            
            'Qubit Flux Bias Voltage': 0 * V 
           }

with qubit_na_experiment.QubitNAExperiment() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    
    # run.sweep('NA Source Power', np.linspace(-70, -30, 41) * dBm, save=True)
    run.sweep('Qubit Flux Bias Voltage', np.linspace(-1.5, 1.5, 121) * V, save=True)