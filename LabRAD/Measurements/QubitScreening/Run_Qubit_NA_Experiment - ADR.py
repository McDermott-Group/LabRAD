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
<<<<<<< HEAD
                    'Interface': 'ADR3',
                    'Variables': {'Temperature': {'Setting': 'Temperatures', 'Stage': 'FAA'}}
=======
                    'Interface': 'Leiden',
                    'Variables': {'Temperature': {'Setting': 'Mix Temperature'}}
>>>>>>> origin/master
                },
                { # Readings entered manually, software parameters
                    'Interface': None,
                    'Variables': []
                }
            ]

# Experiment Information
ExptInfo = {
<<<<<<< HEAD
            'Device Name': '100715A-E4',
            'User': 'Guilhem Ribeill and Chris Wilen',
            'Base Path': 'Z:\mcdermott-group\Data\Flux Biased JPM\ADR Cooldown 110915',
            'Experiment Name': 'FrequencyFluxBias2D',
            'Comments': 'Reflection from measure port of flux biased JPM' 
=======
            'Device Name': 'MH060',
            'User': 'Guilhem Ribeill',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits\Leiden DR 2015-10-22 - Qubits and JPMs',
            'Experiment Name': 'FreqBias2D',
            'Comments': '' 
>>>>>>> origin/master
           }
 
# Experiment Variables
ExptVars = {
<<<<<<< HEAD
            'NA Center Frequency': 4.5 * GHz,
            'NA Frequency Span': 1000 * MHz,
            
            'NA Source Power': -40 * dBm,
            
            'NA Sweep Points': 3201,
            'NA Average Points': 5,
=======
            'NA Center Frequency': 4.914 * GHz,
            'NA Frequency Span': 20 * MHz,
            
            'NA Source Power': -70 * dBm,
            
            'NA Sweep Points': 801,
            'NA Average Points': 200,
>>>>>>> origin/master
            
            'Qubit Flux Bias Voltage': 0 * V 
           }

with qubit_na_experiment.QubitNAExperiment() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    
<<<<<<< HEAD
    # run.sweep('NA Source Power', np.linspace(-70, -30, 41) * dBm, save=True)
=======
    #run.sweep('NA Source Power', np.linspace(-80, -10, 121) * dBm, save=True)
>>>>>>> origin/master
    run.sweep('Qubit Flux Bias Voltage', np.linspace(-2, 2, 201) * V, save=True)