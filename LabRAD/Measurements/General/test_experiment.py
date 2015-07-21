import os.path
if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    SCRIPT_PATH = os.path.dirname(os.getcwd())  # This will be executed when the script is loaded by the labradnode.
else:
    SCRIPT_PATH = os.path.dirname(__file__)     # This will be executed if the script is started by clicking or in a command line.
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
        self.wrap_data_var('Outcome', 'uniform', {'linestyle': 'b-', 'ylim': [0, 1], 'legendlabel': 'Switch. Prob.'})
        
        outcome = np.random.rand(self.variable('Reps'))
        
        run_data = {
                    'Outcome': np.random.rand(self.variable(Reps)),
                    'Outcome Mean': np.mean(outcome),
                    'Outcome Std Dev': np.std(outcome),
                    'Voltage': 10 * V,
                   } 
        
        self.add_expt_var('Actual Reps', len(run_data['Outcome']))
        return run_data, None
        
        
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