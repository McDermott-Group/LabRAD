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

import labrad.units as units

import LabRAD.Measurements.General.experiment as expt


class NISNetworkAnalyzer(expt.Experiment):
    def run_once(self):
        self.set('Bias Voltage')
        
        # Network analyzer variables.
        self.set('NA Source Power')
        self.set('NA Sweep Points')
        self.set('NA Average Points')
        f_center = self.value('NA Center Frequency')
        f_span = self.value('NA Frequency Span')
        self.value('NA Start Frequency', f_center - f_span / 2.)
        self.value('NA Stop Frequency',  f_center + f_span / 2.)
        self.set('NA Start Frequency')
        self.set('NA Stop Frequency')

        self.acknowledge_requests()
        
        self.get('Temperature')
        self.get('S2P')
        
        data = self.acknowledge_request('S2P')
        length = len(data)
        freq = np.empty((length,), dtype=units.Value)
        S11  = np.empty((length,), dtype=units.Value)
        ph11 = np.empty((length,), dtype=units.Value)
        S21  = np.empty((length,), dtype=units.Value)
        ph21 = np.empty((length,), dtype=units.Value)
        S12  = np.empty((length,), dtype=units.Value)
        ph12 = np.empty((length,), dtype=units.Value)
        S22  = np.empty((length,), dtype=units.Value)
        ph22 = np.empty((length,), dtype=units.Value)
        for k in range(length):
            freq[k] = data[k][0]
            S11[k]  = data[k][1]
            ph11[k] = data[k][2]
            S21[k]  = data[k][3]
            ph21[k] = data[k][4]
            S12[k]  = data[k][5]
            ph12[k] = data[k][6]
            S22[k]  = data[k][7]
            ph22[k] = data[k][8]
            
        data = {
                'S11': {
                    'Value': S11,
                    'Dependencies': ['RF Frequency']},
                'S11 Phase': {
                    'Value': ph11,
                    'Dependencies': ['RF Frequency']},
                'S21': {
                    'Value': S21,
                    'Dependencies': ['RF Frequency']},
                'S21 Phase': {
                    'Value': ph21,
                    'Dependencies': ['RF Frequency']},
                'S12': {
                    'Value': S12,
                    'Dependencies': ['RF Frequency']},
                'S12 Phase': {
                    'Value': ph12,
                    'Dependencies': ['RF Frequency']},
                'S22': {
                    'Value': S22,
                    'Dependencies': ['RF Frequency']},
                'S22 Phase': {
                    'Value': ph22,
                    'Dependencies': ['RF Frequency']},
                'RF Frequency': {
                        'Value': freq,
                        'Type': 'Independent'},
                'Temperature': {'Value': self.acknowledge_request('Temperature')}
                }

        return data