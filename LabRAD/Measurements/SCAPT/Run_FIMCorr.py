import os
import numpy as np

from labrad.units import us, ns, mV, V

import fim_corr_expt


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [   {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'mcdermott5125 Board DAC 9',
                          ],
                'mcdermott5125 Board DAC 9':  {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Variables': {  # Default values.
                                'Init Time': {},
                                'Bias Time': {},
                                'Bias Voltage 1': {},
                                'Bias Voltage 2': {}
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
            'Experiment Name': 'FIM Correlation Test 11202015',
            'Comments': '2X 1um Detectors' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 5, # should not exceed ~50,000

            'Init Time': 15000 * us,
            'Bias Time': 10000 * us,
          
            'Bias Voltage 1': 70 * mV, #Into Ch1
            'Bias Voltage 2': 035 * mV,   #Into Ch3
           }

with fim_corr_expt.FIM() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.sweep('Bias Voltage 2', np.linspace(1032, 1034, 11) * mV,
        save=False,
        runs=10000000, max_data_dim=2)