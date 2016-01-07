import os
import numpy as np

from labrad.units import V, GHz, MHz, dB, dBm

import nis_network_analyzer


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   
                { # SIM Voltage Source
                    'Interface': 'SIM928 Voltage Source',
                    'Variables': 'Bias Voltage'
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
                                  'S2P': {'Setting': 'Get S2P',
                                          'Ports': (3, 4)}}
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
            'Device Name': 'Test',
            'User': 'Ivan',
            'Base Path': 'Z:\mcdermott-group\Data\Test',
            'Experiment Name': 'Test',
            'Comments': '' 
           }
 
# Experiment Variables
ExptVars = {
            'NA Center Frequency': 4.82 * GHz,
            'NA Frequency Span': 200 * MHz,
            
            'NA Source Power': -30 * dBm,
            
            'NA Sweep Points': 801,
            'NA Average Points': 16,
            
            'Bias Voltage': 0.1 * V 
           }

with nis_network_analyzer.NISNetworkAnalyzer() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    run.sweep('Bias Voltage', np.linspace(0, 0.1, 5) * V,save=True)          
    #run.sweep('NA Source Power', np.array([-30]) * dBm, save=True)
    #run.sweep('Bias Voltage', np.array([0]) * V, save=True)