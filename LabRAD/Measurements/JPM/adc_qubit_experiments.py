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
import LabRAD.Measurements.General.pulse_shapes as pulse
import LabRAD.Measurements.General.data_processing as dp
from LabRAD.Measurements.General.adc_experiment import ADCExperiment

class ADCQubitReadout(ADCExperiment):
    """
    Read out a qubit connected to a resonator.
    """
    def load_once(self, adc=None, plot_waveforms=False):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        value=self.value('Qubit Frequency') + 
                              self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        value=self.value('Readout Frequency') + 
                              self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse
      
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout Delay')['ns']       # delay from the start of the qubit pulse to the start of the readout pulse
       
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']

        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])
        
        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])
 
        if 'Qubit I' in self.boards.requested_waveforms:            
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms:        
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])
 
        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)
        
        # Delay from the start of the readout pulse to the start of the demodulation.
        ADC_wait_time = self.value('ADC Wait Time')['ns']
        self.boards.set_adc_setting('FilterStartAt', (DAC_ZERO_PAD_LEN +
                ADC_wait_time + QB_time + QBtoRO + RO_time) * units.ns
                , adc)
        self.boards.set_adc_setting('ADCDelay', 0 * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'], sram_length, 0, sram_delay)
                for dac in self.boards.dacs]
        
        ###LOAD####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()


class ADCRamsey(ADCExperiment):
    """
    Ramsey drive and readout of a qubit connected to a resonator.
    """
    def load_once(self, adc=None, plot_waveforms=False):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        value=self.value('Qubit Frequency') + 
                              self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        value=self.value('Readout Frequency') + 
                              self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse
        QB_wait = self.value('Qubit T2 Delay')['ns']                    # delay between two qubit pulses
      
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout Delay')['ns']       # delay from the start of the qubit pulse to the start of the readout pulse
       
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']

        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])
        
        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time*2. + QBtoRO + QB_wait, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time*2. + QBtoRO + QB_wait, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])
 
        if 'Qubit I' in self.boards.requested_waveforms:            
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QB_wait, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms:        
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QB_wait, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])
 
        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)
        
        # Delay from the start of the readout pulse to the start of the demodulation.
        ADC_wait_time = self.value('ADC Wait Time')['ns']
        self.boards.set_adc_setting('ADCDelay', (DAC_ZERO_PAD_LEN +
                ADC_wait_time + QB_time + QBtoRO) * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'], sram_length, 0, sram_delay)
                for dac in self.boards.dacs]
        
        ###LOAD####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()        


class ADCStarkShift(ADCExperiment):
    """
    Read out a qubit connected to a resonator.
    """
    def load_once(self, adc=None, plot_waveforms=False):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        value=self.value('Qubit Frequency') + 
                              self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        value=self.value('Readout Frequency') + 
                              self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #STARK PULSE###############################################################################
        Stark_amp = self.value('Stark Amplitude')['DACUnits']           # amplitude of the Stark pulse
        Stark_time = self.value('Stark Time')['ns']                     # length of the Stark pulse
        
        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse
      
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout Delay')['ns']       # delay from the start of the qubit pulse to the start of the readout pulse
        
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']

        QBtoEnd = QBtoRO + RO_time + DAC_ZERO_PAD_LEN
        
        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoEnd, 0)])
        
        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                pulse.CosinePulse(Stark_time, RO_SB_freq, Stark_amp, 0.0, 0.0),
                                                pulse.DC(QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                pulse.SinePulse(Stark_time, RO_SB_freq, Stark_amp, 0.0, 0.0),
                                                pulse.DC(QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])
 
        if 'Qubit I' in self.boards.requested_waveforms: 
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoEnd, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms: 
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoEnd, 0)])
 
        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('Readout SB Frequency'), adc)

        # Delay from the start of the readout pulse to the start of the demodulation.
        ADC_wait_time = self.value('ADC Wait Time')['ns']
        self.boards.set_adc_setting('ADCDelay', (DAC_ZERO_PAD_LEN +
                ADC_wait_time + QB_time + QBtoRO) * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'], sram_length, 0, sram_delay)
                for dac in self.boards.dacs]
        
        ###LOAD####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()
        

class ADCDemodTest(ADCExperiment):
    """
    Test the ADC demodulation.
    """
    def load_once(self, adc=None, plot_waveforms=False):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        value=self.value('Qubit Frequency') + 
                              self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        value=self.value('Readout Frequency') + 
                              self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse
      
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout Delay')['ns']       # delay from the start of the qubit pulse to the start of the readout pulse
       
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']

        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])
        
        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])
 
        if 'Qubit I' in self.boards.requested_waveforms:            
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms:        
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])
 
        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', self.value('ADC Demod Frequency'), adc)
        
        # Delay from the start of the readout pulse to the start of the demodulation.
        ADC_wait_time = self.value('ADC Wait Time')['ns']
        self.boards.set_adc_setting('ADCDelay', (DAC_ZERO_PAD_LEN +
                ADC_wait_time + QB_time + QBtoRO) * units.ns, adc)

        mems = [seq.mem_simple(self.value('Init Time')['us'], sram_length, 0, sram_delay)
                for dac in self.boards.dacs]
        
        ###RUN#####################################################################################
        result = self.boards.load(dac_srams, mems)
        self.acknowledge_requests()

        
class ADCCavityJPM(ADCExperiment):
    """
    Probe a resonator that is driven by a switching JPM with a ADC.
    """
    def load_once(self, adc=None, plot_waveforms=False):
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('RF Attenuation')                                      # RF attenuation
        self.set('RF Power')                                            # RF power
        if self.value('RF Frequency') is not None:
            if self.value('RF SB Frequency') is not None:               # RF frequency
                self.set('RF Frequency',
                        self.value('RF Frequency') + 
                        self.value('RF SB Frequency'))
            else:
                self.set('RF Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')
      
        #JPM VARIABLES#############################################################################
        JPM_FPT = self.value('Fast Pulse Time')['ns']                   # length of the DAC pulse
        JPM_FPA = self.value('Fast Pulse Amplitude')['DACUnits']        # amplitude of the DAC pulse
        JPM_FPW = self.value('Fast Pulse Width')['ns']                  # DAC pulse rise time 
        
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']
        
        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
                
        waveforms = {}
        if 'JPM Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + JPM_smoothed_FP.size, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)

        ###SET BOARDS PROPERLY#####################################################################
        self.boards.set_adc_setting('DemodFreq', -self.value('RF SB Frequency'), adc)
        
        # Delay from the start of the readout pulse to the start of the demodulation.
        ADC_wait_time = self.value('ADC Wait Time')['ns']
        self.boards.set_adc_setting('ADCDelay', (DAC_ZERO_PAD_LEN +
                ADC_wait_time) * units.ns, adc)
        
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