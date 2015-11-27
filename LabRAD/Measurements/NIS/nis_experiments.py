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


#experiment calling order:
# init_expt()
# load_once()
# run_once()
# exit_expt()

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

import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq
import LabRAD.Measurements.General.pulse_shapes as pulse
import LabRAD.Measurements.General.data_processing as dp
from   LabRAD.Measurements.General.adc_experiment import ADCExperiment


class NISReadout(ADCExperiment):
    """
    NIS experiment.
    """
    def load_once(self, plot_waveforms=False):
        #RF VARIABLES###################################################
        self.set('RF Power')
        self.set('RF Frequency')

        ReadoutDelay = self.value('Bias to Readout Delay')['ns']
      
        ###WAVEFORMS####################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']

        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN, 0)])
 
        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)
        
        self.boards.set_adc_setting('ADCDelay', (DAC_ZERO_PAD_LEN +
                self.value('ADC Wait Time')['ns']) * units.ns)
                
        b2ro_time = self.value('Bias to Readout Delay')['us']
        bias_time = self.value('NIS Bias Time')['us']
        if bias_time > b2ro_time:
            bias_time = bias_time - b2ro_time
        else:
            bias_time = 0

        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()

        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': self.value('NIS Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': b2ro_time})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})        
        mem_lists[0].append({'Type': 'Delay', 'Time': bias_time})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        
        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[1].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('NIS Bias Time')['us']})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]    

        ###LOAD#########################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()