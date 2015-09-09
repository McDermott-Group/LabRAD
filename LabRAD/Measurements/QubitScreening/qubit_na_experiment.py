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

    def run_once(self):
        if self.value('Qubit Flux Bias Voltage') is not None:
            self.send_request('Qubit Flux Bias Voltage')
        
        # Network analyzer variables.
        NA_centerFreq = self.value('NA Center Frequency')
        NA_freqSpan = self.value('NA Frequency Span')
        NA_srcPower = self.value('NA Source Power')
        NA_freqPoints = self.value('NA Frequency Points')
        NA_avgPoints = self.value('NA Average Points')
            
        # Select a network analyzer.
        na_server = self.cxn.agilent_5230a_network_analyzer()
        dev = na_server.list_devices()
        na_server.select_device(dev[0][0])
        
        start_freq  = NA_centerFreq - NA_freqSpan/2.
        stop_freq   = NA_centerFreq + NA_freqSpan/2.
        
        frequency = np.linspace(start_freq['GHz'], stop_freq['GHz'],
                NA_freqPoints) * GHz
        
        na_server.start_frequency(start_freq)
        na_server.stop_frequency(stop_freq)
        na_server.sweep_points(NA_freqPoints)
        na_server.average_points(NA_avgPoints)
        na_server.source_power(NA_srcPower)
        
        if NA_avgPoints > 1:
            na_server.average_mode(True)
            
        self.acknowledge_requests()
        if self.get_interface('Temperature') is not None:
            self.send_request('Temperature')
        
        S21data = na_server.get_trace()
        na_server.deselect_device()
        
        data = {
                'Transmission': {
                    'Value': self.strip_units(S21data)*dB,
                    'Dependencies': ['RF Frequency']},
                'RF Frequency': {
                        'Value': frequency,
                        'Type': 'Independent'}
                }
        
        if self.get_interface('Temperature') is not None:
            data['Temperature'] = {'Value': self.acknowledge_request('Temperature')}
            
        return data