import os
if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    # This is executed when the script is loaded by the labradnode.
    SCRIPT_PATH = os.path.dirname(os.getcwd())
else:
    # This is executed if the script is started by clicking or
    # from a command line.
    SCRIPT_PATH = os.path.dirname(__file__)
LABRAD_PATH = os.path.join(SCRIPT_PATH.rsplit('LabRAD', 1)[0])
import sys
if LABRAD_PATH not in sys.path:
    sys.path.append(LABRAD_PATH)
 
import numpy as np

import labrad.units as units

from LabRAD.Measurements.General.experiment import Experiment


class SimpleTestExperiment(Experiment):
    """
    Mock-up a simple experiment.
    """
    def run_once(self):
        outcome = np.random.rand(self.variable('Reps'))
        run_data = {
                    'Outcome': {'Value': outcome,
                                'Mean': 'Outcome Mean',
                                'Std Dev': 'Outcome Std',
                                'Dependencies': ['Runs'],
                                'Distribution': 'uniform',
                                'Prefereances': {'linestyle': 'b-', 
                                                 'ylim': [0, 1], 
                                                 'legendlabel': 'Switch. Prob.'}
                               },
                    'Outcome Mean': {'Value': np.mean(outcome),
                                     'Prefereances': {'linestyle': 'r-',
                                                      'ylim': [0, 1], 
                                                      'legendlabel': 'Mean'}
                                    },
                    'Outcome Std': {'Value': np.mean(outcome)},
                    'Voltage': {'Value': 10 * units.V},
                    'Runs': {'Value': np.linspace(1, self.variable('Reps'), self.variable('Reps')),
                             'Type': 'Independent'}
                   }
         
        self.add_var('Actual Reps', len(run_data['Outcome']))
        return run_data
        
# List of the experiment resources. Simply uncomment/comment the devices that should be used/unused.
# However, 'Resource': 'LabRAD Server' should never be left out.
Resources = [
                { # External readings.
 'Resource': 'Manual Record',
                    'Variables': ['Temperature']
                },
                { # Extra experiment parameters.
                    'Resource': 'Software Parameters',
                    'Variables': ['Reps', 'Variable 1', 'Variable 2'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'Test Device',
            'User': 'Test User',
            'Base Path': 'Z:\mcdermott-group\Data\Test',
            'Experiment Name': 'SimpleExperiment',
            'Comments': '' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 10000, # should not exceed ~50,000
          
            'Variable 1': 10 * units.V,
            'Variable 2': 100 * units.us,
            'Variable 3': -10 * units.V,
  
            'Temperature': 14.2 * units.mK
           }

with SimpleTestExperiment() as expt:
    
    expt.set_experiment(ExptInfo, Resources, ExptVars) 
    
    expt.sweep('Variable 1', np.linspace(0.1, 0.3, 101) * units.V, 
                save=True, print_data=['Outcome Mean'], plot_data=['Outcome Mean'])