# Copyright (C) 2015 Ted Thorbeck
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
import LabRAD.Measurements.General.pulse_shapes as pulse
import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq

DAC_ZERO_PAD_LEN = 10

class FIM(expt.Experiment):
    """
    Test.
    """
    def run_once(self):
        requested_waveforms = [settings[ch] for settings in
                self.boards.dac_settings for ch in ['DAC A', 'DAC B']]

        waveforms = {}
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()
       
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[0].append({'Type': 'Timer', 'Time': 0})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]
        
        self.send_request('Temperature')
        P = self.boards.load_and_run(dac_srams, mems, self.value('Reps'))
        self.add_var('Actual Reps', len(P[0]))
        
        
        

        return {
                'Temperature': {'Value': self.acknowledge_request('Temperature')}
               }
               
               
               
               
               
               
               
               
               
               
               
               
               
               
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