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
import matplotlib.pyplot as plt

import labrad.units as units

import LabRAD.Measurements.General.experiment as expt
import LabRAD.Measurements.General.pulse_shapes as pulse
import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq
import data_processing as dp


class JPMExperiment(expt.Experiment):
    def _plot_histogram(self, data, number_of_devices=1, 
            preamp_timeout=1253):
        if number_of_devices == 0:
            return
        data = np.array(data)
        plt.figure(3)
        plt.ion()
        plt.clf()
        if number_of_devices == 1: 
            plt.hist(data[0,:], bins=preamp_timeout, range=(1, preamp_timeout-1),
                color='b')
        elif number_of_devices == 2:
            plt.hist(data[0,:], bins=preamp_timeout, range=(1, preamp_timeout-1),
                color='r', label='JPM A')
            plt.hist(data[1,:], bins=preamp_timeout, range=(1, preamp_timeout-1),
                color='b', label='JPM B')
            plt.legend()
        elif number_of_devices > 2:
            raise Exception('Histogram plotting for more than two ' +
            'devices is not implemented.')
        plt.xlabel('Timing Information [Preamp Time Counts]')
        plt.ylabel('Counts')
        plt.xlim(0, preamp_timeout) 
        plt.draw()
        plt.pause(0.05)
        
    def run_once(self, histogram=False):
        ###RUN#####################################################################################
        self.get('Temperature')
        P = self.boards.run(self.value('Reps'))

        ###EXTRA EXPERIMENT PARAMETERS TO SAVE#####################################################
        self.add_var('Actual Reps', len(P[0]))
        
        preamp_timeout = self.value('Preamp Timeout')['PreAmpTimeCounts']
        threshold = self.value('Threshold')['PreAmpTimeCounts']
        
        ###DATA POST-PROCESSING####################################################################
        if histogram:
            self._plot_histogram(P, 1, preamp_timeout)
            print('Maximum timing counts: ' + str(np.max(P) * units.PreAmpTimeCounts) + '.')

        P = np.array(P)
        t_mean, t_std = dp.mean_time(P, 0, preamp_timeout)

        ###DATA STRUCTURE##########################################################################
        return {
                'Switching Probability': {
                    'Value': dp.prob(P, 0, threshold),
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'b-',
                        'ylim': [0, 1],
                        'legendlabel': 'Switch. Prob.'}},
                'Detection Time': {
                    'Value': t_mean * units.PreAmpTimeCounts,
                    'Distribution': 'normal',
                    'Preferences': {
                        'linestyle': 'r-', 
                        'ylim': [0, preamp_timeout]}},
                'Detection Time Std Dev': {
                    'Value': t_std * units.PreAmpTimeCounts},
                'Temperature': {'Value': self.acknowledge_request('Temperature')}
               }


class JPMQubitReadout(JPMExperiment):
    """
    Read the qubit state with a JPM by applying a read-out and 
    a displacement (reset) pulses.
    """
    def load_once(self, plot_waveforms=False):
        #RF VARIABLES##############################################################################
        self.set('RF Attenuation')                                      # RF attenuation
        self.set('RF Power')                                            # RF power
        if self.value('RF Frequency') is not None:
            if self.value('RF SB Frequency') is not None:               # RF frequency
                self.set('RF Frequency',
                        self.value('RF Frequency') + 
                        self.value('RF SB Frequency'))
            else:
                self.set('RF Frequency')

        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        self.value('Qubit Frequency') + 
                        self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        self.value('Readout Frequency') + 
                        self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        RO_phase = self.value('Readout Phase')['rad']                   # readout pulse phase
        Disp_amp = self.value('Displacement Amplitude')['DACUnits']     # amplitude of the displacement pulse
        Disp_time = self.value('Displacement Time')['ns']               # length of the displacement pulse time
        Disp_phase = self.value('Displacement Phase')['rad']            # displacement pulse phase
        ROtoD_offset = self.value('Readout to Displacement Offset')['DACUnits'] # zero offset between readout and reset pulses
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse

        #JPM VARIABLES#############################################################################
        JPM_FPT = self.value('Fast Pulse Time')['ns']                   # length of the DAC pulse
        JPM_FPA = self.value('Fast Pulse Amplitude')['DACUnits']        # amplitude of the DAC pulse
        JPM_FPW = self.value('Fast Pulse Width')['ns']                  # DAC pulse rise time 
        
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout')['ns']             # delay between the end of the qubit pulse and the start of the readout pulse
        ROtoD = self.value('Readout to Displacement')['ns']             # delay between the end of readout pulse and the start of the displacement pulse
        DtoFP = self.value('Displacement to Fast Pulse')['ns']          # delay between the end of the displacement pulse and the start of the fast pulse

        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']
        
        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        FPtoEnd = max(0, DtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN
        
        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])
        
        if 'JPM Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + DtoFP, 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(max(0, -DtoFP - JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])      

        if 'Qubit I' in self.boards.requested_waveforms: 
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms: 
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase, 0),
                                                pulse.CosinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase, 0),
                                                pulse.CosinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase, 0),
                                                pulse.DC(FPtoEnd, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase, 0),
                                                pulse.SinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase, 0),
                                                pulse.SinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase, 0),
                                                pulse.DC(FPtoEnd, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()

        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1,
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': 3})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Mode': 'Fine',
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        # mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Slow'})
        mem_lists[0].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        
        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('Init Time')['us'] +
                                      self.value('Bias Time')['us']})
        mem_lists[1].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[1].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]
               
        ###LOAD#################################################################################### 
        self.boards.load(dac_srams, mems)
        self.acknowledge_requests()


class JPMStarkShift(JPMExperiment):
    """
    Stark shift experiment with a JPM.
    """
    def load_once(self, plot_waveforms=False):
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        self.value('Qubit Frequency') + 
                        self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        self.value('Readout Frequency') + 
                        self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #STARK PULSE###############################################################################
        Stark_amp = self.value('Stark Amplitude')['DACUnits']           # amplitude of the Stark pulse
        Stark_time = self.value('Stark Time', 'ns')['ns']               # length of the Stark pulse
        
        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        RO_phase = self.value('Readout Phase')['rad']                   # readout pulse phase
                
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse

        #JPM VARIABLES#############################################################################
        JPM_FPT = self.value('Fast Pulse Time')['ns']                   # length of the DAC pulse
        JPM_FPA = self.value('Fast Pulse Amplitude')['DACUnits']        # amplitude of the DAC pulse
        JPM_FPW = self.value('Fast Pulse Width')['ns']                  # DAC pulse rise time 
        
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout')['ns']             # delay between the end of the qubit pulse and the start of the readout pulse
        ROtoFP = self.value('Readout to Fast Pulse')['ns']              # delay between the end of readout pulse and the start of the fast pulse 
        
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']
        
        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        QBtoEnd = (QBtoRO + max(0, -RO_time - ROtoFP) + RO_time + 
                max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN)
        
        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoEnd, 0)])
        
        if 'JPM Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoRO + max(0, RO_time + ROtoFP), 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(max(0, -JPM_smoothed_FP.size - ROtoFP) + DAC_ZERO_PAD_LEN, 0)])     

        if 'Qubit I' in self.boards.requested_waveforms: 
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoEnd, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms: 
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoEnd, 0)])

        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                pulse.CosinePulse(Stark_time, RO_SB_freq, Stark_amp, 0.0, 0.0),
                                                pulse.DC(QBtoRO + max(0, -RO_time - ROtoFP), 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                pulse.SinePulse(Stark_time, RO_SB_freq, Stark_amp, 0.0, 0.0),
                                                pulse.DC(QBtoRO + max(0, -RO_time - ROtoFP), 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()

        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1,
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': 3})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Mode': 'Fine',
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        # mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Slow'})
        mem_lists[0].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        
        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('Init Time')['us'] +
                                      self.value('Bias Time')['us']})
        mem_lists[1].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[1].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]
               
        ###LOAD#################################################################################### 
        self.boards.load(dac_srams, mems)
        self.acknowledge_requests()

        
class JPMTwoPhotonSpectroscopy(JPMExperiment):
    """
    JPM spectroscopy with two RF tones.
    """
    def load_once(self, plot_waveforms=False):
        #RF1 VARIABLES#############################################################################
        self.set('RF1 Attenuation')                                     # RF1 attenuation
        self.set('RF1 Power')                                           # RF1 power
        if self.value('RF1 Frequency') is not None:
            if self.value('RF1 SB Frequency') is not None:              # RF1 frequency
                self.set('RF1 Frequency',
                        self.value('RF1 Frequency') + 
                        self.value('RF1 SB Frequency'))
            else:
                self.set('RF1 Frequency')
    
        #RF2 VARIABLES#############################################################################
        self.set('RF2 Attenuation')                                     # RF2 attenuation
        self.set('RF2 Power')                                           # RF2 power
        if self.value('RF2 Frequency') is not None:
            if self.value('RF2 SB Frequency') is not None:              # RF2 frequency
                self.set('RF2 Frequency',
                        self.value('RF2 Frequency') + 
                        self.value('RF2 SB Frequency'))
            else:
                self.set('RF2 Frequency')

        #RF VARIABLES##############################################################################
        RF1_SB_freq = self.value('RF1 SB Frequency')['Hz']              # RF1 sideband frequency
        RF1_amp = self.value('RF1 Amplitude')['DAC units']              # amplitude of the sideband modulation
        RF1_time = self.value('RF1 Time')['ns']                         # length of the RF1 pulse
    
        #QUBIT DRIVE VARIABLES###############################################################################
        RF2_SB_freq = self.value('RF2 SB Frequency')['Hz']              # RF2 sideband frequency
        RF2_amp = self.value('RF2 Amplitude')['DAC units']              # amplitude of the sideband modulation
        RF2_time = self.value('RF2 Time')['ns']                         # length of the RF2 pulse
        
        #TIMING VARIABLES####################################################################################
        RF1toRF2 = self.value('RF1 to RF2')['ns']                       # delay between the start of the RF1 pulse and the start of the RF2 pulse
        RF2toFP = self.value('RF2 to Fast Pulse')['ns']                 # delay between the start of the RF2 pulse and the start of the fast pulse

        #JPM VARIABLES#############################################################################
        JPM_FPT = self.value('Fast Pulse Time')['ns']                   # length of the DAC pulse
        JPM_FPA = self.value('Fast Pulse Amplitude')['DACUnits']        # amplitude of the DAC pulse
        JPM_FPW = self.value('Fast Pulse Width')['ns']                  # DAC pulse rise time 
        
        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']
        
        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        total_time = max(max(0, -RF1toRF2) + RF1_time, 
                         max(0, RF1toRF2) + RF2_time, 
                         max(0, RF1toRF2)+ RF2toFP + JPM_smoothed_FP.size)
        
        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + total_time, 0)])
        
        if 'JPM Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2) + RF2toFP, 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(total_time - max(0, RF1toRF2) - RF2toFP - JPM_smoothed_FP.size)])  

        if 'RF1 I' in self.boards.requested_waveforms: 
            waveforms['RF1 I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, -RF1toRF2), 0),
                                            pulse.CosinePulse(RF1_time, RF1_SB_freq, RF1_amp, 0, 0),
                                            pulse.DC(total_time - max(0, -RF1toRF2) - RF1_time + DAC_ZERO_PAD_LEN, 0)])

        if 'RF1 Q' in self.boards.requested_waveforms: 
            waveforms['RF1 Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, -RF1toRF2), 0),
                                            pulse.SinePulse(RF1_time, RF1_SB_freq, RF1_amp, 0, 0),
                                            pulse.DC(total_time - max(0, -RF1toRF2) - RF1_time + DAC_ZERO_PAD_LEN, 0)])

        if 'RF2 I' in self.boards.requested_waveforms:
            waveforms['RF2 I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2), 0),
                                            pulse.CosinePulse(RF2_time, RF2_SB_freq, RF2_amp, 0.0, 0.0),
                                            pulse.DC(total_time - max(0, RF1toRF2) - RF2_time + DAC_ZERO_PAD_LEN, 0)])

        if 'RF2 Q' in self.boards.requested_waveforms:
            waveforms['RF2 Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2), 0),
                                            pulse.SinePulse(RF2_time, RF2_SB_freq, RF2_amp, 0.0, 0.0),
                                            pulse.DC(total_time - max(0, RF1toRF2) - RF2_time + DAC_ZERO_PAD_LEN, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()

        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1,
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': 3})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Mode': 'Fine',
                'Voltage': self.value('Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        # mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Slow'})
        mem_lists[0].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        
        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('Init Time')['us'] +
                                      self.value('Bias Time')['us']})
        mem_lists[1].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[1].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]
               
        ###LOAD#################################################################################### 
        self.boards.load(dac_srams, mems)
        self.acknowledge_requests()


class DoubleJunctionJPMQubitReadout(JPMExperiment):
    """
    Read out a qubit connected to a resonator with a readout and 
    a displacement (reset) pulse. This class is developed for
    the experiments with two-junction JPMs.
    """
    def run_once(self, histogram=False, plot_waveforms=False):
        #BIAS VOLTAGES#############################################################################
        BV_out = self.value('Bias Voltage')['V']                        # output bias voltage
        BV_in = self.value('Input Bias Voltage')['V']                   # input bias voltage
        
        if (BV_in + BV_out < 0 or 
                BV_in + BV_out >= self.value('Max Bias Voltage')['V']):
            return {
                    'Switching Probability': {
                        'Value': 0,
                        'Distribution': 'binomial',
                        'Preferences':  {
                            'linestyle': 'b-',
                            'ylim': [0, 1],
                            'legendlabel': 'Switch. Prob.'}},
                    'Detection Time': {
                        'Value': -1 * units.PreAmpTimeCounts,
                        'Distribution': 'normal',
                        'Preferences': {
                            'linestyle': 'r-', 
                            'ylim': [0, self.ghz_fpga_boards.consts['PREAMP_TIMEOUT']]}},
                    'Detection Time Std Dev': {
                        'Value': 0 * units.PreAmpTimeCounts},
                    'Temperature': {'Value': np.nan * units.mK}
                   }
        
        BV_step = abs(self.value('Bias Voltage Step')['V'])             # bias voltage step
        BV_step_time = self.value('Bias Voltage Step Time')['us']       # bias voltage step time 
    
        #QUBIT VARIABLES###########################################################################
        self.set('Qubit Attenuation')                                   # qubit attenuation
        self.set('Qubit Power')                                         # qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # qubit frequency
                self.set('Qubit Frequency',
                        self.value('Qubit Frequency') + 
                        self.value('Qubit SB Frequency'))
            else:
                self.set('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.set('Readout Attenuation')                                 # readout attenuation
        self.set('Readout Power')                                       # readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # readout frequency
                self.set('Readout Frequency',
                        self.value('Readout Frequency') + 
                        self.value('Readout SB Frequency'))
            else:
                self.set('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        self.set('Qubit Flux Bias Voltage')

        #CAVITY DRIVE (READOUT) VARIABLES##########################################################
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']          # readout sideband frequency
        RO_amp = self.value('Readout Amplitude')['DACUnits']            # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                      # length of the readout pulse
        RO_phase = self.value('Readout Phase')['rad']                   # readout pulse phase
        Disp_amp = self.value('Displacement Amplitude')['DACUnits']     # amplitude of the displacement pulse
        Disp_time = self.value('Displacement Time')['ns']               # length of the displacement pulse time
        Disp_phase = self.value('Displacement Phase')['rad']            # displacement pulse phase
        ROtoD_offset = self.value('Readout to Displacement Offset')['DACUnits'] # zero offset between readout and reset pulses
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']            # qubit sideband frequency
        QB_amp = self.value('Qubit Amplitude')['DACUnits']              # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                        # length of the qubit pulse

        #JPM VARIABLES#############################################################################
        JPM_FPT = self.value('Fast Pulse Time')['ns']                   # length of the DAC pulse
        JPM_FPA = self.value('Fast Pulse Amplitude')['DACUnits']        # amplitude of the DAC pulse
        if BV_in + BV_out < 0:
            JPM_FPA = -JPM_FPA
        JPM_FPW = self.value('Fast Pulse Width')['ns']                  # DAC pulse rise time 
        
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout')['ns']             # delay between the end of the qubit pulse and the start of the readout pulse
        ROtoD = self.value('Readout to Displacement')['ns']             # delay between the end of readout pulse and the start of displacement pulse
        DtoFP = self.value('Displacement to Fast Pulse')['ns']          # delay between the end of the displacement pulse and the start of the fast pulse

        ###WAVEFORMS###############################################################################
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']
        
        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        FPtoEnd = max(0, DtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN
        
        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])
        
        if 'JPM Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + DtoFP, 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(max(0, -DtoFP - JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])      

        if 'Qubit I' in self.boards.requested_waveforms: 
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Qubit Q' in self.boards.requested_waveforms: 
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Readout I' in self.boards.requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase, 0),
                                                pulse.CosinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase, 0),
                                                pulse.CosinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase, 0),
                                                pulse.DC(FPtoEnd, 0)])

        if 'Readout Q' in self.boards.requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase, 0),
                                                pulse.SinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase, 0),
                                                pulse.SinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase, 0),
                                                pulse.DC(FPtoEnd, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()
       
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': 0})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        
        if BV_in * BV_out < 0:
            n_in = int(np.floor(abs(BV_in / BV_step)))
            n_out = int(np.floor(abs(BV_out / BV_step)))
            step_in = np.sign(BV_in) * BV_step
            step_out = np.sign(BV_out) * BV_step
            for k in np.linspace(1, min(n_in, n_out), min(n_in, n_out)):
                mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': k * step_out})
                mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': k * step_in})
                mem_lists[0].append({'Type': 'Delay', 'Time': BV_step_time})
        
        if abs(BV_in) < abs(BV_out):
            mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': BV_out})
            mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': BV_in})
        else:
            mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': BV_in})
            mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': BV_out})
 
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[0].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': 0})

        mem_lists[1].append({'Type': 'Delay', 'Time': self.value('Init Time')['us'] +
                                      self.value('Bias Time')['us']})
        mem_lists[1].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[1].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})

        mems = [seq.mem_from_list(mem_list) for mem_list in mem_lists]
        
        ###RUN#####################################################################################
        self.acknowledge_requests()
        self.get('Temperature')
        P = self.boards.load_and_run(dac_srams, mems, self.value('Reps'))

        ###EXTRA EXPERIMENT PARAMETERS TO SAVE#####################################################
        self.add_var('Actual Reps', len(P[0]))
        
        preamp_timeout = self.value('Preamp Timeout')['PreAmpTimeCounts']
        threshold = self.value('Threshold')['PreAmpTimeCounts']
        
        ###DATA POST-PROCESSING####################################################################
        if histogram:
            self._plot_histogram(P, 1)

        P = np.array(P)
        t_mean, t_std = dp.mean_time(P, 0, preamp_timeout)

        ###DATA STRUCTURE##########################################################################
        return {
                'Switching Probability': {
                    'Value': dp.prob(P, 0, threshold),
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'b-',
                        'ylim': [0, 1],
                        'legendlabel': 'Switch. Prob.'}},
                'Detection Time': {
                    'Value': t_mean * units.PreAmpTimeCounts,
                    'Distribution': 'normal',
                    'Preferences': {
                        'linestyle': 'r-', 
                        'ylim': [0, preamp_timeout]}},
                'Detection Time Std Dev': {
                    'Value': t_std * units.PreAmpTimeCounts},
                'Temperature': {'Value': self.acknowledge_request('Temperature')}
               }