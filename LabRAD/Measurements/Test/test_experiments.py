# Copyright (C) 2015 Guilhem Ribeill, Ivan Pechenezhskiy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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

import LabRAD.Measurements.General.experiment as expt

class TestExpt(expt.Experiment):
    """
    Read out a qubit connected to a resonator with a readout and a displacement (reset) pulse.
    """
    def run_once(self, histogram=False, plot_waveforms=False):
        data = {
                'Switching Probability': {
                    'Value': np.random.rand(),
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'b-',
                        'ylim': [0, 1],
                        'legendlabel': 'Switch. Prob.'}},
                'Detection Time': {
                    'Value': 10 * np.random.rand() * units.s,
                    'Distribution': 'normal',
                    'Preferences': {
                        'linestyle': 'r-', 
                        'ylim': [0, 20]}},
                'Detection Time Std Dev': {
                    'Value': 4 * np.random.rand() * units.s}
               } 
        self.add_var('Actual Reps', self.value('Reps'))
        
        return data