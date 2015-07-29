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

import LabRAD.Measurements.General.experiment as expt
import LabRAD.Measurements.General.pulse_shapes as pulse

import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq

import data_processing

DAC_ZERO_PAD_LEN = 20


class JPMExperiment(expt.Experiment):
    def _plot_histogram(self, data, number_of_devices=1, 
            pream_timeout=1253):
        if number_of_devices == 0:
            return
        data = np.array(data)
        plt.figure(3)
        plt.ion()
        plt.clf()
        if number_of_devices == 1: 
            plt.hist(data[0, :], bins=500, range=(0, pream_timeout),
                color='b')
        elif number_of_devices == 2:
            plt.hist(data[0, :], bins=500, range=(0, pream_timeout),
                color='b', label='JPM A')
            plt.hist(data[1, :], bins=500, range=(0, pream_timeout),
                color='r', label='JPM B')
            plt.legend()
        elif number_of_devices > 2:
            raise Exception('Histogram plotting for more than two ' +
            'devices is not implemented.')
        plt.xlabel('Timing Information [counts]')
        plt.ylabel('Counts')
        plt.draw()


class JPMQubitReadoutWithReset(JPMExperiment):
    """
    Read out a qubit connected to a resonator with a readout and a displacement (reset) pulse.
    """
    def run_once(self, histogram=False, plot_waveforms=False):
        #RF VARIABLES##############################################################################
        if self.value('RF Attenuation') is not None:
            self.send_request('RF Attenuation')                         # RF attenuation
        if self.value('RF Power') is not None:
            self.send_request('RF Power')                               # RF power
        if self.value('RF Frequency') is not None:
            if self.value('RF SB Frequency') is not None:               # RF frequency
                self.send_request('RF Frequency', False,                
                        self.value('RF Frequency') + 
                        self.value('RF SB Frequency'))
            else:
                self.send_request('RF Frequency')

        #QUBIT VARIABLES###########################################################################
        if self.value('Qubit Attenuation') is not None:
            self.send_request('Qubit Attenuation')                      # Qubit attenuation
        if self.value('Qubit Power') is not None:
            self.send_request('Qubit Power')                            # Qubit power
        if self.value('Qubit Frequency') is not None:
            if self.value('Qubit SB Frequency') is not None:            # Qubit frequency
                self.send_request('Qubit Frequency', False,
                        self.value('Qubit Frequency') + 
                        self.value('Qubit SB Frequency'))
            else:
                self.send_request('Qubit Frequency')
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        if self.value('Readout Attenuation') is not None:
            self.send_request('Readout Attenuation')                    # Readout attenuation
        if self.value('Readout Power') is not None:
            self.send_request('Readout Power')                          # Readout power
        if self.value('Readout Frequency') is not None:
            if self.value('Readout SB Frequency') is not None:          # Readout frequency
                self.send_request('Readout Frequency', False,
                        self.value('Readout Frequency') + 
                        self.value('Readout SB Frequency'))
            else:
                self.send_request('Readout Frequency')

        #DC BIAS VARIABLES#########################################################################
        if self.value('Qubit Flux Bias Voltage') is not None:
            self.send_request('Qubit Flux Bias Voltage', False)
        
        ###EXPERIMENT VARIABLES USED BY PERMANENTLY PRESENT DEVICES################################
        # Experiment variables that used by DC Rack, DAC and ADC boards should be defined here.
           
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
      
        #JPM A VARIABLES########################################################################### 
        JPM_bias = self.value('Bias Voltage')['V']                      # height of the FastBias pulse   
        JPM_FPT = self.value('Fast Pulse Time')['ns']                   # length of the DAC pulse
        JPM_FPA = self.value('Fast Pulse Amplitude')['DACUnits']        # amplitude of the DAC pulse
        JPM_FPW = self.value('Fast Pulse Width')['ns']                  # DAC pulse rise time 
        
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout')['ns']             # delay between the end of the qubit pulse and the start of the readout pulse
        ROtoD = self.value('Readout to Displacement')['ns']             # delay between the end of readout pulse and the start of displacement pulse
        DtoFP = self.value('Displacement to Fast Pulse')['ns']          # delay between the end of the displacement pulse and the start of the fast pulse

        ###WAVEFORMS###############################################################################
        fpga = self.ghz_fpga_boards
        requested_waveforms = [settings[ch] for settings in
                fpga.dac_settings for ch in ['DAC A', 'DAC B']]

        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        FPtoEnd = max(0, DtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN
        
        waveforms = {};
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])
        
        if 'JPM Fast Pulse' in requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + DtoFP, 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(max(0, -DtoFP - JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])      

        if 'Qubit I' in requested_waveforms: 
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Qubit Q' in requested_waveforms: 
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Readout I' in requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase, 0),
                                                pulse.CosinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase, 0),
                                                pulse.CosinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase, 0),
                                                pulse.DC(FPtoEnd, 0)])

        if 'Readout Q' in requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase, 0),
                                                pulse.SinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase, 0),
                                                pulse.SinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase, 0),
                                                pulse.DC(FPtoEnd, 0)])

        for idx, settings in enumerate(fpga.dac_settings):
            for channel in ['DAC A', 'DAC B']:
                if fpga.dac_settings[idx][channel] not in waveforms:
                    raise expt.ExperimentDefinitionError("'" + 
                        str(fpga.dacs[idx]) +
                        "' setting '" + str(channel) + "': '" +
                        fpga.dac_settings[idx][channel] +
                        "' could not be recognized. The allowed '" +
                        str(channel) + "' values are 'JPM Fast Pulse'," + 
                        "'Readout I', 'Readout Q', 'Qubit I', 'Qubit Q'," +
                        " and 'None'.")

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in requested_waveforms],
                    ['r', 'g', 'b', 'k'], requested_waveforms)

        sram_length = len(waveforms[fpga.dac_settings[0]['DAC A']])
        sram_delay = np.ceil(sram_length / 1000)
        # Create a memory command list.
        # The format is described in Servers.Instruments.GHzBoards.command_sequences.
        if 'FO1 FastBias Firmware Version' in fpga.dac_settings[0]:
            mem_list1 = [{'Type': 'Firmware', 'Channel': 1, 
                          'Version': fpga.dac_settings[0]['FO1 FastBias Firmware Version']}]
        else:
            mem_list1 = []
        mem_list1 = mem_list1 + [
            {'Type': 'Bias', 'Channel': 1, 'Voltage': 0},
            {'Type': 'Delay', 'Time': self.value('Init Time')['us']},
            {'Type': 'Bias', 'Channel': 1, 'Voltage': self.value('Bias Voltage')['V']},
            {'Type': 'Delay', 'Time': self.value('Bias Time')['us']},
            {'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay},
            {'Type': 'Timer', 'Time': self.value('Measure Time')['us']},
            {'Type': 'Bias', 'Channel': 1, 'Voltage': 0}]

        mem_list2 = [
            {'Type': 'Delay', 'Time': (self.value('Init Time')['us'] + 
                                       self.value('Bias Time')['us'])},
            {'Type': 'SRAM', 'Start': 0, 'Length': sram_length, 'Delay': sram_delay},
            {'Type': 'Timer', 'Time': self.value('Measure Time')['us']}]
        
        dac_srams = [seq.waves2sram(waveforms[fpga.dac_settings[k]['DAC A']], 
                                    waveforms[fpga.dac_settings[k]['DAC B']])
                                    for k, dac in enumerate(fpga.dacs)]
        mems1 = seq.mem_from_list(mem_list1)
        mems2 = seq.mem_from_list(mem_list2)
        
        ###RUN#####################################################################################
        self.acknowledge_requests()
        P = fpga.load_and_run(dac_srams, [mems1, mems2], self.value('Reps'))

        ###DATA POST-PROCESSING####################################################################
        if histogram:
            self._plot_histogram(P, 1)

        preamp_timeout = fpga.consts['PREAMP_TIMEOUT']
        t_mean, t_std = data_processing.mean_time_from_array(P, preamp_timeout)
        
        ###DATA STRUCTURE##########################################################################
        data = {
                'Switching Probability': {
                    'Value': data_processing.prob_from_array(P, self.value('Threshold')),
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
                    'Value': t_std * units.PreAmpTimeCounts}
               } 
        
        ###EXTRA EXPERIMENT PARAMETERS TO SAVE#####################################################
        self.add_var('Actual Reps', len(P[0]))
        
        return data