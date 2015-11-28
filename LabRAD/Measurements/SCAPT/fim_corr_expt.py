# Copyright (C) 2015
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

import LabRAD.Measurements.General.experiment as expt
import LabRAD.Measurements.General.waveform as wf
import LabRAD.Servers.Instruments.GHzBoards.mem_sequences as ms


class FIM(expt.Experiment):
    """
    Test experiment.
    """
    def run_once(self):
        ###WAVEFORMS####################################################        
        waveforms, offset = wf.wfs_dict(self.boards.consts['DAC_ZERO_PAD_LEN'])
        
        dac_srams, sram_length = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###MEMORY COMMAND LISTS#########################################
        # The format is described in Servers.Instruments.GHzBoards.mem_sequences.
        mem_seqs = self.boards.init_mem_lists()

        mem_seqs[0].bias(1, voltage=0, mode='Fast')
        mem_seqs[0].bias(2, voltage=0, mode='Fast')

        mem_seqs[0].delay(self.value('Init Time'))

        mem_seqs[0].bias(1, voltage=self.value('Bias Voltage 1'))
        mem_seqs[0].bias(2, voltage=self.value('Bias Voltage 2'))
        
        mem_seqs[0].delay(self.value('Bias Time'))
        
        mem_seqs[0].bias(1, voltage=0)
        mem_seqs[0].bias(2, voltage=0)
        
        mem_seqs[0].sram(sram_length=sram_length, sram_start=0)
        mem_seqs[0].timer(0)

        mems = [mem_seq.sequence() for mem_seq in mem_seqs]

        ###RUN##########################################################
        self.get('Temperature')
        P = self.boards.load_and_run(dac_srams, mems, self.value('Reps'))
        self.add_var('Actual Reps', len(P[0]))
                
        return {
                'Temperature': {'Value': self.acknowledge_request('Temperature')}
               }