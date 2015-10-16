import numpy as np

from labrad.units import (us, ns, V, GHz, MHz, rad, mK, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

import test_experiments as test


Resources = [   { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': [
                                  'Dummy 1',
                                  'Dummy 2',
                                  'Dummy 3',
                                  'Actual Reps',
                                 ],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'None',
            'User': 'Sam',
            'Base Path': 'Z:\mcdermott-group\Data\Test',
            'Experiment Name': 'Dummy',
            'Comments': 'A Comment' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 1000,
            'Dummy 1': 1 * V,
            'Dummy 2': 2 * MHz,
            'Dummy 3': 3
           }

with test.TestExpt() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    run.sweep('Dummy 1', np.linspace(.1, 0.3, 201) * V,
            save=True, print_data=['Switching Probability'], plot_data=['Switching Probability'])

    run.sweep(['Dummy 2', 'Dummy 3'], 
              [np.linspace(-0.75, 0.75, 301) * MHz, np.linspace(-0.75, 0.75, 101)],
              save=True, print_data=['Switching Probability'])