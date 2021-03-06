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
                                ' GPIB Bus - GPIB0::2::INSTR::SIM900::4'),
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
                    'Interface': 'ADR3',
                    'Variables': {'Temperature': {'Setting': 'Temperatures', 'Stage': 'FAA'}}
                },
                { # Readings entered manually, software parameters
                    'Interface': None,
                    'Variables': []
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'MH070B',
            'User': 'Guilhem Ribeill and Chris Wilen',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\ADR3 111615',
            'Experiment Name': 'FreqPower2D',
            'Comments': 'Reflection from measure port of flux biased JPM' 
           }
 
# Experiment Variables
ExptVars = {
            'NA Center Frequency': 4.82 * GHz,
            'NA Frequency Span': 200 * MHz,
            
            'NA Source Power': -40 * dBm,
            
            'NA Sweep Points': 801,
            'NA Average Points': 64,
            
            'Qubit Flux Bias Voltage': 0 * V 
           }

with qubit_na_experiment.QubitNAExperiment() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    
    run.sweep('NA Source Power', np.linspace(-70, -30, 41) * dBm, save=True)
    #run.sweep('Qubit Flux Bias Voltage', np.array([0]) * V, save=True)