# Copyright (C) 2015 Guilhem Ribeill
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

from labrad.units import V, GHz, MHz, mK, dB, dBm

import LabRAD.Measurements.General.experiment as expt


class QubitNAExperiment(expt.Experiment):
    """Qubit spectroscopy with a network analyzer."""
    def run_once(self):
        self.send_request('Qubit Flux Bias Voltage')
        
        # Network analyzer variables.
        self.send_request('NA Source Power')
        self.send_request('NA Frequency Points')
        self.send_request('NA Average Points')
        if self.value('NA Average Points') > 1:
            self.value('Average Mode', True, output=False)
            self.send_request('Average Mode')
        
        NA_centerFreq = self.value('NA Center Frequency')
        NA_freqSpan = self.value('NA Frequency Span')
        self.value('NA Start Frequency', NA_centerFreq - NA_freqSpan / 2.,
            output=False)
        self.value('NA Stop Frequency', NA_centerFreq + NA_freqSpan / 2.,
            output=False)
        self.send_request('Start Frequency')
        self.send_request('Stop Frequency')

        frequency = np.linspace(start_freq['GHz'], stop_freq['GHz'],
        NA_freqPoints) * GHz

        self.acknowledge_requests()
        self.send_request('Temperature')
        self.send_request('Trace')
        
        data = {
                'Transmission': {
                    'Value': self.strip_units(self.acknowledge_request('Trace')) * dB,
                    'Dependencies': ['RF Frequency']},
                'RF Frequency': {
                        'Value': frequency,
                        'Type': 'Independent'}
                }
        
        if self.get_interface('Temperature') is not None:
            data['Temperature'] = {'Value': self.acknowledge_request('Temperature')}
            
        return data