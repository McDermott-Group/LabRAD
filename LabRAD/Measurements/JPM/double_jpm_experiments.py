# Copyright (C) 2015 Ivan Pechenezhskiy
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

import LabRAD.Measurements.General.pulse_shapes as pulse
import LabRAD.Measurements.General.server_interfaces as si
import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq
import data_processing as dp
from jpm_qubit_experiments import JPMExperiment


class DoubleJPMCorrelation(JPMExperiment):
    """
    Double JPM experiment.
    """
    def run_once(self, histogram=False, plot_waveforms=False):
        #DC BIAS VARIABLES#########################################################################
        if (self.value('DC Bias Voltage') is not None and
            abs(self.value('DC Bias Voltage')['mV']) <= 30):
            self.set('DC Bias Voltage')
        else:
            raise Exception("'DC Bias Voltage' should not exceed 30 mV.")

        #RF VARIABLES##############################################################################
        self.set('RF Attenuation')                             # RF attenuation
        self.set('RF Power')                                   # RF power
        if self.value('RF Frequency') is not None:
            if self.value('RF SB Frequency') is not None:               # RF frequency
                self.set('RF Frequency',                
                        self.value('RF Frequency') + 
                        self.value('RF SB Frequency'))
            else:
                self.set('RF Frequency')
 
        #RF DRIVE VARIABLES########################################################################
        RF_SB_freq = self.value('RF SB Frequency')['GHz']               # readout sideband frequency
        RF_amp = self.value('RF Amplitude')['DACUnits']                 # amplitude of the sideband modulation
        RF_time = self.value('RF Time')['ns']                           # length of the readout pulse
        
        #JPM A VARIABLES###########################################################################
        JPMA_FPT = self.value('JPM A Fast Pulse Time')['ns']            # length of the DAC pulse
        JPMA_FPA = self.value('JPM A Fast Pulse Amplitude')['DACUnits'] # amplitude of the DAC pulse
        JPMA_FW = self.value('JPM A Fast Pulse Width')['ns']            # DAC pulse rise time 

        #JPM B VARIABLES###########################################################################
        JPMB_FPT = self.value('JPM B Fast Pulse Time')['ns']            # length of the DAC pulse
        JPMB_FPA = self.value('JPM B Fast Pulse Amplitude')['DACUnits'] # amplitude of the DAC pulse
        JPMB_FW = self.value('JPM B Fast Pulse Width')['ns']            # DAC pulse rise time
        
        #TIMING VARIABLES##########################################################################
        RFtoFP_A = self.value('RF to JPM A Fast Pulse Delay')['ns']     # delay between the start of the RF pulse and the start of the JPM A pulse
        JPMdelay = self.value('JPM A to JPM B Fast Pulse Delay')['ns']  # delay between the JPM A and JPM B fast pulses (can be negative, i.e. the JPM B pulse is ahead of the JPM A one)
   
        ###WAVEFORMS###############################################################################
        JPMA_FP = pulse.GaussPulse(JPMA_FPT, JPMA_FW, JPMA_FPA)
        JPMB_FP = pulse.GaussPulse(JPMB_FPT, JPMB_FW, JPMB_FPA)
        
        DAC_ZERO_PAD_LEN = self.boards.consts['DAC_ZERO_PAD_LEN']['ns']
        waveforms = {}
        if 'None' in self.boards.requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + RF_time, 0)])
        
        # Set up fast pulses.
        if 'JPM A Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM A Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + RFtoFP_A, 0),
                                                      JPMA_FP,
                                                      pulse.DC(RF_time - RFtoFP_A - JPMA_FP.size + DAC_ZERO_PAD_LEN, 0)])

        if 'JPM B Fast Pulse' in self.boards.requested_waveforms:
            waveforms['JPM B Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + RFtoFP_A + JPMdelay, 0),
                                                      JPMB_FP,
                                                      pulse.DC(RF_time - RFtoFP_A - JPMdelay - JPMB_FP.size + DAC_ZERO_PAD_LEN, 0)])

        # Set up RF pulse.
        if 'RF I' in self.boards.requested_waveforms:      
            waveforms['RF I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                           pulse.CosinePulse(RF_time, RF_SB_freq, RF_amp, 0.0, 0.0),
                                           pulse.DC(DAC_ZERO_PAD_LEN, 0)]) 

        if 'RF Q' in self.boards.requested_waveforms:
            waveforms['RF Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                           pulse.SinePulse(RF_time, RF_SB_freq, RF_amp, 0.0, 0.0),
                                           pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        dac_srams, sram_length, sram_delay = self.boards.process_waveforms(waveforms)

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in self.boards.requested_waveforms],
                    ['r', 'g', 'b', 'k'], self.boards.requested_waveforms)
        
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        mem_lists = self.boards.init_mem_lists()

        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Init Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1,
                'Voltage': self.value('JPM A Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 2,
                'Voltage': self.value('JPM B Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Mode': 'Fine',
                'Voltage': self.value('JPM A Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Mode': 'Fine',
                'Voltage': self.value('JPM B Bias Voltage')['V']})
        mem_lists[0].append({'Type': 'Delay', 'Time': self.value('Bias Time')['us']})
        mem_lists[0].append({'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 2, 'Voltage': 0, 'Mode': 'Fast'})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Slow'})
        mem_lists[0].append({'Type': 'Timer', 'Time': self.value('Measure Time')['us']})
        mem_lists[0].append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0, 'Mode': 'Fast'})
        
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
        
        ###DATA POST-PROCESSING####################################################################
        preamp_timeout = self.boards.consts['PREAMP_TIMEOUTS'][self.value('Measure Time')['us']]
        self.add_var('Preamp Timeout', preamp_timeout * units.PreAmpTimeCounts
        min_threshold = self.value('Min Threshold')['PreAmpTimeCounts']
        max_threshold = self.value('Max Threshold')['PreAmpTimeCounts']

        if histogram:
            self._plot_histogram(P, 2, preamp_timeout)
        
        P = np.array(P)
        ta_mean, ta_std = dp.mean_time(P[0,:], 0, preamp_timeout)
        tb_mean, tb_std = dp.mean_time(P[1,:], 0, preamp_timeout)
        dt_mean, dt_std = dp.mean_time_diff(P, 0, preamp_timeout)
        
        outcomes = dp.outcomes(P, min_threshold, max_threshold)
        print('Maximum timing value = ' + str(np.max(P)) + ' PreAmpTimeCounts')
        n = float(np.shape(outcomes)[1])

        return {
                'Pa': {
                    'Value': dp.prob(P[0,:], min_threshold, max_threshold),
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'r-',
                        'ylim': [0, 1],
                        'legendlabel': 'JPM A Switch. Prob.'}},
                'Pb': {
                    'Value': dp.prob(P[1], min_threshold, max_threshold),
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'b-',
                        'ylim': [0, 1],
                        'legendlabel': 'JPM B Switch. Prob.'}},
                'Corr Coef': {
                    'Value': dp.corr_coef_from_outcomes(outcomes),
                    'Preferences': {
                        'linestyle': 'g-',
                        'ylim': [-1, 1],
                        'legendlabel': 'Corr. Coefficient'}},
                'JPM A Detection Time': {
                    'Value': ta_mean * units.PreAmpTimeCounts,
                    'Distribution': 'normal',
                    'Preferences': {
                        'linestyle': 'r-', 
                        'ylim': [0, preamp_timeout]}},
                'JPM A Detection Time Std Dev': {
                    'Value': ta_std * units.PreAmpTimeCounts},
                'JPM B Detection Time': {
                    'Value': tb_mean * units.PreAmpTimeCounts,
                    'Distribution': 'normal',
                    'Preferences': {
                        'linestyle': 'b-', 
                        'ylim': [0, preamp_timeout]}},
                'JPM B Detection Time Std Dev': {
                    'Value': tb_std * units.PreAmpTimeCounts},
                'Detection Time Diff': {
                    'Value': dt_mean * units.PreAmpTimeCounts,
                    'Distribution': 'normal',
                    'Preferences': {
                        'linestyle': 'k-', 
                        'ylim': [-preamp_timeout, preamp_timeout]}},
                'Detection Time Diff Std Dev': {
                    'Value': dt_std * units.PreAmpTimeCounts},
                'P00': {
                    'Value': float(((1 - outcomes[0,:]) * (1 - outcomes[1,:])).sum()) / n,
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'k-',
                        'ylim': [0, 1],
                        'legendlabel': 'P00'}},
                'P01': {
                    'Value': float(((1 - outcomes[0,:]) * outcomes[1,:]).sum()) / n,
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'g-',
                        'ylim': [0, 1],
                        'legendlabel': 'P01'}},
                'P10': {
                    'Value': float((outcomes[0,:] * (1 - outcomes[1,:])).sum()) / n,
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'c-',
                        'ylim': [0, 1],
                        'legendlabel': 'P10'}},
                'P11': {
                    'Value': float((outcomes[0,:] * outcomes[1,:]).sum()) / n,
                    'Distribution': 'binomial',
                    'Preferences':  {
                        'linestyle': 'm-',
                        'ylim': [0, 1],
                        'legendlabel': 'P11'}},
                'Temperature': {'Value': self.acknowledge_request('Temperature')},
                # 'JPM A Timing Data': {
                    # 'Value': P[0,:] * units.PreAmpTimeCounts,
                    # 'Dependencies': ['Repetition Index'],
                    # 'Preferences': {
                        # 'linestyle': 'r-', 
                        # 'ylim': [0, preamp_timeout]}},
                # 'JPM B Timing Data': {
                    # 'Value': P[1,:] * units.PreAmpTimeCounts,
                    # 'Dependencies': ['Repetition Index'],
                    # 'Preferences': {
                        # 'linestyle': 'b-', 
                        # 'ylim': [0, preamp_timeout]}},
                # 'Repetition Index': {
                        # 'Value': np.linspace(1, len(P[0]), len(P[0])),
                        # 'Type': 'Independent'}
               }