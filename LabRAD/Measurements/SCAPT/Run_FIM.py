import os
import numpy as np

from labrad.units import (us, ns, mV, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import fim_experiment


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
            'Reps': 301, # should not exceed ~50,000

            'Init Time': 500 * us,
            'Bias Time': 10000 * us,
          
            'Bias Voltage': 100 * mV,
           }

with fim_experiment.FIM() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.sweep('Bias Voltage', np.linspace(48, 50, 11) * mV,
        save=True, print_data=['Temperature'], plot_data=['Temperature'],
        runs=100, max_data_dim=1)   
