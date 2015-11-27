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

import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq
import LabRAD.Measurements.General.waveform as wf
import LabRAD.Measurements.General.data_processing as dp
from LabRAD.Measurements.General.adc_experiment import ADCExperiment


class ADCQubitReadout(ADCExperiment):
    """
    Read out a qubit connected to a resonator.
    """
    def load_once(self, adc=None):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        self.set('Qubit Frequency', self.value('Qubit Frequency') +     # qubit frequency
                self.value('Qubit SB Frequency'))
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        self.set('Readout Frequency', self.value('Readout Frequency') + # readout frequency
                self.value('Readout SB Frequency'))

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        ###WAVEFORMS###############################################################################
        QB_I, QB_Q = wf.Harmonic(amplitude = self.value('Qubit Amplitude'),
                                 frequency = self.value('Qubit SB Frequency'),
                                 start     = 0,
                                 duration  = self.value('Qubit Time'))
        
        RO_I, RO_Q = wf.Harmonic(amplitude = self.value('Readout Amplitude'),
                                 frequency = self.value('Readout SB Frequency'),
                                 start     = QB_I.after(self.value('Qubit Drive to Readout Delay')),
                                 duration  = self.value('Readout Time'))
        
        waveforms, offset = wf.wfs_dict(self.boards.consts['DAC_ZERO_PAD_LEN'],
                wf.Waveform('Readout I', RO_I), wf.Waveform('Readout Q', RO_Q),
                wf.Waveform('Qubit I',   QB_I), wf.Waveform('Qubit Q',   QB_Q))

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)

        # Delay between the end of the readout pulse to the start of the demodulation.
        self.boards.set_adc_setting('FilterStartAt', (offset +
                RO_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'],
                sram_length, 0, sram_delay) for dac in self.boards.dacs]
        
        ###LOAD####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()


class ADCRamsey(ADCExperiment):
    """
    Ramsey drive and readout of a qubit connected to a resonator.
    """
    def load_once(self, adc=None):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        self.set('Qubit Frequency', self.value('Qubit Frequency') +     # qubit frequency
                self.value('Qubit SB Frequency'))
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        self.set('Readout Frequency', self.value('Readout Frequency') + # readout frequency
                self.value('Readout SB Frequency'))

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')
      
        ###WAVEFORMS###############################################################################
        QB1_I, QB1_Q = wf.Harmonic(amplitude = self.value('Qubit Amplitude'),
                                   frequency = self.value('Qubit SB Frequency'),
                                   start     = 0,
                                   duration  = self.value('Qubit Time'))
                                   
        QB2_I, QB2_Q = wf.Harmonic(amplitude = self.value('Qubit Amplitude'),
                                   frequency = self.value('Qubit SB Frequency'),
                                   start     = QB1_I.after(self.value('Qubit T2 Delay')),
                                   duration  = self.value('Qubit Time'))
        
        RO_I, RO_Q = wf.Harmonic(amplitude = self.value('Readout Amplitude'),
                                 frequency = self.value('Readout SB Frequency'),
                                 start     = QB2_I.after(self.value('Qubit Drive to Readout Delay')),
                                 duration  = self.value('Readout Time'))
        
        waveforms, offset = wf.wfs_dict(self.boards.consts['DAC_ZERO_PAD_LEN'],
                wf.Waveform('Readout I', RO_I),
                wf.Waveform('Readout Q', RO_Q),
                wf.Waveform('Qubit I', QB1_I, QB2_I),
                wf.Waveform('Qubit Q', QB1_Q, QB2_Q))

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)

        # Delay between the end of the readout pulse to the start of the demodulation.
        self.boards.set_adc_setting('FilterStartAt', (offset +
                RO_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'], sram_length, 0, sram_delay)
                for dac in self.boards.dacs]
        
        ###LOAD####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()        


class ADCStarkShift(ADCExperiment):
    """
    Read out a qubit connected to a resonator.
    """
    def load_once(self, adc=None):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        self.set('Qubit Frequency', self.value('Qubit Frequency') +     # qubit frequency
                self.value('Qubit SB Frequency'))
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        self.set('Readout Frequency', self.value('Readout Frequency') + # readout frequency
                self.value('Readout SB Frequency'))

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')
      
        ###WAVEFORMS###############################################################################
        Stark_I, Stark_Q = wf.Harmonic(amplitude = self.value('Stark Amplitude'),
                                       frequency = self.value('Readout SB Frequency'),
                                       start     = 0),
                                       duration  = self.value('Stark Time'))
        
        QB_I, QB_Q = wf.Harmonic(amplitude = self.value('Qubit Amplitude'),
                                 frequency = self.value('Qubit SB Frequency'),
                                 duration  = self.value('Qubit Time'),
                                 end       = Stark_I.end)

        RO_I, RO_Q = wf.Harmonic(amplitude = self.value('Readout Amplitude'),
                                 frequency = self.value('Readout SB Frequency'),
                                 start     = QB_I.after(self.value('Qubit Drive to Readout Delay')),
                                 duration  = self.value('Readout Time'))
        
        waveforms, offset = wf.wfs_dict(self.boards.consts['DAC_ZERO_PAD_LEN'],
                wf.Waveform('Readout I', Stark_I, RO_I),
                wf.Waveform('Readout Q', Stark_Q, RO_Q),
                wf.Waveform('Qubit I', QB_I),
                wf.Waveform('Qubit Q', QB_Q))

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)

        # Delay between the end of the readout pulse to the start of the demodulation.
        self.boards.set_adc_setting('FilterStartAt', (offset +
                RO_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'], sram_length, 0, sram_delay)
                for dac in self.boards.dacs]
        
        ###LOAD####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()

        
class ADCCavityJPM(ADCExperiment):
    """
    Probe a resonator that is driven by a switching JPM with a ADC.
    """
    def load_once(self, adc=None):
        #RF VARIABLES##############################################################################
        self.set('RF Attenuation')                                      # RF attenuation
        self.set('RF Power')                                            # RF power
        self.set('RF Frequency', self.value('RF Frequency') +           # RF frequency
                self.value('RF SB Frequency'))
                self.set('RF Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        ###WAVEFORMS###############################################################################
        JPM = wf.Square(amplitude = self.value('Fast Pulse Amplitude'),
                        start     = 0,
                        duration  = self.value('Fast Pulse Time'))
        
        waveforms, offset = wf.wfs_dict(self.boards.consts['DAC_ZERO_PAD_LEN'],
                wf.Waveform('JPM Fast Pulse', JPM))
        
        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        # wf.plot_wfs(waveforms, waveforms.keys())

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)

        # Delay between the end of the readout pulse to the start of the demodulation.
        self.boards.set_adc_setting('FilterStartAt', (offset +
                RO_I.end + self.value('ADC Wait Time')['ns']) * units.ns, adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()
        
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1,
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[0].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})

        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('Init Time')['us'] +
                                      self.value('Bias Time')['us']})
        mem_lists[1].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[1].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]
        
        ###LOAD####################################################################################
        self.acknowledge_requests()
        self.boards.load(dac_srams, mems)