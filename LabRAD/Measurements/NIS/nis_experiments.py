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

import LabRAD.Servers.Instruments.GHzBoards.mem_sequences as ms
import LabRAD.Measurements.General.waveform as wf
import LabRAD.Measurements.General.data_processing as dp
from   LabRAD.Measurements.General.adc_experiment import ADCExperiment


class NISReadout(ADCExperiment):
    """
    NIS experiment.
    """
    def load_once(self, adc=None):
        #RF VARIABLES###################################################
        self.set('RF Power')
        self.set('RF Frequency', self.value('RF Frequency') +
            self.value('RF SB Frequency'))
      
        ###WAVEFORMS####################################################
        RF_I, RF_Q = wf.Harmonic(amplitude = self.value('RF Amplitude'),
                                 frequency = self.value('RF SB Frequency'),
                                 start     = 0,
                                 duration  = self.value('RF Time'))        
        
        wfs, offset = wf.wfs_dict(wf.Waveform('RF I', RF_I),
                                  wf.Waveform('RF Q', RF_Q),
                                  min_length=self.boards.consts['DAC_ZERO_PAD_LEN'])
        dac_srams, sram_length = self.boards.process_waveforms(wfs)

        # wf.plot_wfs(wfs, wfs.keys())

        ###SET BOARDS PROPERLY##########################################
        self.boards.set_adc_setting('DemodFreq', -self.value('RF SB Frequency'), adc)
        
        self.boards.set_adc_setting('ADCDelay', (offset +
                RF_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        self.boards.set_adc_setting('FilterStartAt', 0 * units.ns, adc)

        ###MEMORY COMMAND LISTS#########################################
        mem_seqs = self.boards.init_mem_lists()

        mem_seqs[0].bias(1, voltage=0, mode='Fast')
        mem_seqs[0].delay(self.value('Init Time'))
        mem_seqs[0].bias(1, voltage=self.value('NIS Bias Voltage'))
        mem_seqs[0].delay(self.value('Bias to RF Delay'))
        mem_seqs[0].sram(sram_length=sram_length, sram_start=0)
        mem_seqs[0].delay(max(0, self.value('NIS Bias Time') -
                                 self.value('Bias to RF Delay')))
        mem_seqs[0].bias(1, voltage=0)

        for k in range(1, len(self.boards.dacs)):
            mem_seqs[k].delay(self.value('Init Time') +
                              self.value('Bias to RF Delay'))
            mem_seqs[k].sram(sram_length=sram_length, sram_start=0)
            mem_seqs[k].delay(max(0, self.value('NIS Bias Time') -
                         self.value('Bias to RF Delay')))
        mems = [mem_seq.sequence() for mem_seq in mem_seqs]  

        ###LOAD#########################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()