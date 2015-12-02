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
import LabRAD.Servers.Instruments.GHzBoards.mem_sequences as ms
import LabRAD.Measurements.General.waveform as wf
import LabRAD.Measurements.General.data_processing as dp
from LabRAD.Measurements.General.adc_experiment import ADCExperiment


class ADCTestDemodulation(ADCExperiment):
    """
    Read out a qubit connected to a resonator.
    """
    def load_once(self, adc=None):
    
        #RF DRIVE (READOUT) VARIABLES###################################
        self.set('RF Attenuation')
        self.set('RF Power')
        self.set('RF Frequency', self.value('RF Frequency') +
                self.value('RF SB Frequency'))

        
        RF_Q, RF_I = wf.Harmonic(amplitude = self.value('RF Amplitude'),
                                 frequency = self.value('RF SB Frequency'),
                                 phase     = self.value('RF Phase'),
                                 start     = 0,
                                 duration  = self.value('RF Time'))
        SW = wf.ECLDuringPulses(RF_Q)
        
        waveforms, offset = wf.wfs_dict(wf.Waveform('RF I', RF_I),
                                        wf.Waveform('RF Q', RF_Q),
                                        wf.Waveform('Switch', SW),
                      min_length=self.boards.consts['DAC_ZERO_PAD_LEN'])

        dac_srams, sram_length = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###SET BOARDS PROPERLY##########################################
        self.boards.set_adc_setting('DemodFreq', -self.value('ADC Demod Frequency'), adc)
        #self.boards.set_adc_setting('FilterLength', self.value('RF Time'), adc) 
        self.boards.set_adc_setting('FilterStartAt', offset * units.ns, adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)

        # self.boards.set_adc_setting('ADCDelay', (offset +
                 # RO_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        # self.boards.set_adc_setting('FilterStartAt', 0 * units.ns, adc)

        mems = [ms.simple_sequence(self.value('Init Time'), sram_length, 0)
                for dac in self.boards.dacs]
        
        ###LOAD#########################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()
        
class ADCTestLogSpiral(ADCExperiment):
    """
    Read out a qubit connected to a resonator.
    """
    def load_once(self, adc=None):
    
        #RF DRIVE (READOUT) VARIABLES###################################
        self.set('RF Attenuation')
        self.set('RF Power')
        self.set('RF Frequency', self.value('RF Frequency') +
                self.value('RF SB Frequency'))

        a = self.value('RF Amplitude')
        b = 2.*np.pi/45. #spiral pitch, 8 degrees
        
        t = self.value('Spiral')
        Qamp = a * np.exp(b * t)  * np.cos(t) / np.exp(20. * b)
        Iamp = a * np.exp(b * t)  * np.sin(t) / np.exp(20. * b)

        RF_I = wf.Cosine(amplitude = Qamp,
                         frequency = self.value('RF SB Frequency'),
                         phase = self.value('RF Phase'),
                         start = 0,
                         duration = self.value('RF Time'))
                         
        RF_Q = wf.Cosine(amplitude = Iamp,
                         frequency = self.value('RF SB Frequency'),
                         phase = self.value('RF Phase'),
                         start = 0,
                         duration = self.value('RF Time'))
        
        waveforms, offset = wf.wfs_dict(wf.Waveform('RF I', RF_I),
                                        wf.Waveform('RF Q', RF_Q),
                      min_length=self.boards.consts['DAC_ZERO_PAD_LEN'])

        dac_srams, sram_length = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###SET BOARDS PROPERLY##########################################
        self.boards.set_adc_setting('DemodFreq', -self.value('ADC Demod Frequency'), adc)
        #self.boards.set_adc_setting('FilterLength', self.value('RF Time'), adc) 
        self.boards.set_adc_setting('FilterStartAt', offset * units.ns, adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)

        # self.boards.set_adc_setting('ADCDelay', (offset +
                 # RO_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        # self.boards.set_adc_setting('FilterStartAt', 0 * units.ns, adc)

        mems = [ms.simple_sequence(self.value('Init Time'), sram_length, 0)
                for dac in self.boards.dacs]
        
        ###LOAD#########################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()