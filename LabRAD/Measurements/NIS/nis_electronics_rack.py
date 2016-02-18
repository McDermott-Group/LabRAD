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

import LabRAD.Servers.Instruments.GHzBoards.mem_sequences as ms
import LabRAD.Measurements.General.waveform as wf
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
        self.boards.set_adc_setting('DemodFreq',
                -self.value('RF SB Frequency'), adc)

        self.boards.set_adc_setting('ADCDelay', (offset +
                self.value('ADC Wait Time')['ns']) * units.ns, adc)
        
        self.boards.set_adc_setting('FilterLength',
                self.value('ADC Filter Length'), adc)

        ###MEMORY COMMAND LISTS#########################################
        mem_seqs = self.boards.init_mem_lists()

        # 'Fine':    0...2.5 V, slew rate ~ 32 us
        # 'Fast': -2.5...2.5 V, slew rate ~ 4 us
        # 'Slow': -2.5...2.5 V, slew rate ~ 33 us (or 12 us)
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

        
class NISReadoutRelaxation(ADCExperiment):
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
        self.boards.set_adc_setting('DemodFreq',
                -self.value('RF SB Frequency'), adc)

        self.boards.set_adc_setting('ADCDelay', (offset +
                self.value('ADC Wait Time')['ns']) * units.ns, adc)
        
        self.boards.set_adc_setting('FilterLength',
                self.value('ADC Filter Length'), adc)

        ###MEMORY COMMAND LISTS#########################################
        mem_seqs = self.boards.init_mem_lists()

        # 'Fine':    0...2.5 V, slew rate ~ 32 us
        # 'Fast': -2.5...2.5 V, slew rate ~ 4 us
        # 'Slow': -2.5...2.5 V, slew rate ~ 33 us (or 12 us)
        rf_delay = self.value('Bias to RF Delay')
        bias_time = self.value('NIS Bias Time')

        mem_seqs[0].bias(1, voltage=0, mode='Fast')
        mem_seqs[0].delay(self.value('Init Time'))
        mem_seqs[0].bias(1, voltage=self.value('NIS Bias Voltage'))
        calib = self.value('Calibration Coefficient')
        if rf_delay <= bias_time - calib * self.value('RF Time'):
            bias_time = bias_time - calib * self.value('RF Time')
            mem_seqs[0].delay(rf_delay)
            mem_seqs[0].sram(sram_length=sram_length, sram_start=0)
            if bias_time > rf_delay:
                mem_seqs[0].delay(bias_time - rf_delay)
            mem_seqs[0].bias(1, voltage=0)
        else:
            mem_seqs[0].delay(bias_time)
            mem_seqs[0].bias(1, voltage=0)
            if bias_time < rf_delay:
                mem_seqs[0].delay(rf_delay - bias_time)
            mem_seqs[0].sram(sram_length=sram_length, sram_start=0)

        for k in range(1, len(self.boards.dacs)):
            mem_seqs[k].delay(self.value('Init Time'))
            if rf_delay <= bias_time:
                bias_time = bias_time - calib * self.value('RF Time')
                mem_seqs[k].delay(rf_delay)
                mem_seqs[k].sram(sram_length=sram_length, sram_start=0)
                if bias_time > rf_delay:
                    mem_seqs[k].delay(bias_time - rf_delay)
            else:
                mem_seqs[k].delay(bias_time)
                if bias_time < rf_delay:
                    mem_seqs[k].delay(rf_delay - bias_time)
                mem_seqs[k].sram(sram_length=sram_length, sram_start=0)

        mems = [mem_seq.sequence() for mem_seq in mem_seqs]  

        ###LOAD#########################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()