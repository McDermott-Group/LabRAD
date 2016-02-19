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
            'Device Name': 'CutinGND01182016',
            'User': 'Umesh',
            'Base Path': 'Z:\mcdermott-group\Data\NIS Junctions\NIScutinGNDSiox',
            'Experiment Name': 'CWMeasureNA12',
            'Comments': '1D with 100kiloohm Bias' 
           }
 
# Experiment Variables
ExptVars = {
            'NA Center Frequency': 5.168 * GHz,
            'NA Frequency Span': 7 * MHz,
            
            'NA Source Power': -40 * dBm,
            
            'NA Sweep Points': 360,
            'NA Average Points': 2500,
            
            'Bias Voltage': 1.5 * V 
           }

with nis_network_analyzer.NISNetworkAnalyzer() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars) 
    # run.sweep('Bias Voltage', np.linspace(0, 1, 11) * V, save=True)
    # run.sweep('NA Source Power', np.linspace(-45,-50,3) * dBm, save=True)
    # run.sweep('NA Source Power', np.linspace(-52.5,-55,2) * dBm, save=True)      
    run.sweep('NA Source Power', np.array([-40]) * dBm, save=True)
    # run.sweep('Bias Voltage', np.array([0]) * V, save=True)
    # bias_range = np.linspace(1.5,5,8) * V
    # bias_range = np.linspace(0.01,0.05,3) * V
    # for voltage in bias_range:
        # run.value('Bias Voltage', voltage)
        # run.sweep('NA Source Power', 
                 # np.linspace(-25,-40,4) * dBm,save=True)
    
    # for voltage in bias_range:
        # run.value('Bias Voltage', voltage)
        # run.sweep('NA Source Power', 
                 # np.linspace(-45,-55,5) * dBm,save=True)
    
    
    
    
    # voltage = np.linspace(0.95,1,2)
    # for v in voltage:
        # run.sweep('Bias Voltage', np.array([v]) * V, save=True) 