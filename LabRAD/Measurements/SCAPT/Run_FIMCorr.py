import os
import numpy as np

from labrad.units import us, ns, mV, V

import fim_corr_expt


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
            'Reps': 301, # should not exceed ~50,000

            'Init Time': 15000 * us,
            'Bias Time': 10000 * us,
          
            'Bias Voltage 1': 1000 * mV,
            'Bias Voltage 2': 1 * mV,
           }

with fim_corr_expt.FIM() as run:
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.sweep('Bias Voltage 1', np.linspace(2200, 2500, 11) * mV,
        save=False, print_data=['Temperature'], plot_data=['Temperature'],
        runs=1, max_data_dim=1)