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

import os.path
if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    SCRIPT_PATH = os.path.dirname(os.getcwd())  # This will be executed when the script is loaded by the labradnode.
else:
    SCRIPT_PATH = os.path.dirname(__file__)     # This will be executed if the script is started by clicking or in a command line.
LABRAD_PATH = os.path.join(SCRIPT_PATH.rsplit('LabRAD', 1)[0])
import sys
if LABRAD_PATH not in sys.path:
    sys.path.append(LABRAD_PATH)

import numpy as np

import LabRAD.Measurements.General.experiment as expt
import LabRAD.Servers.GHzBoards.ghz_fpga_control as dac
import LabRAD.Servers.GHzBoards.pulse_shapes as pulse

G = float(10**9)
M = float(10**6)

DAC_ZERO_PAD_LEN = 20
PREAMP_TIMEOUT = 1253

#######################################################################
############## HOW TO USE WAVEFORM CLASS ##############################
#
# import path.to.wavepulse as wp
# import path.to.waveform as wf
#
# r = wp.WavePulse("block", 0 ,3, 2, 10, None)
# s = wp.WavePulse("sine", 0, 10, 0.25, None, 10)
# w = wf.WaveForm(r,s, ...) #as many as you want
#
# w.getArr() gives you the array that you need
# r.start, r.end and r.duration return exactly what you think
#
# wavepulses are declared with (type, start, amplitude, frequency, end=None, duration=None)
# where type is either "block", "sine", "cosine", or "gauss"
#######################################################################
#######################################################################


class JPMTwoPhotonExpt(expt.Experiment):
    """
    Read out a RF2 connected to a resonator with a RF1 and a displacement (reset) pulse.
    """
    def RunOnce(self, Histogram=False, PlotWaveforms=False):
        ###DATA VARIABLES####################################################################################
        #####################################################################################################
        # Units for data variables as well as plotting preferences can be defined here.
        # Example: self._WrapDataVar('P',  '', 'binomial', ' {'name': 'Probability', 'linestyle': 'b-', 'linewidth': 2, 'legendlabel': 'Prob.', 'ylim': [0, 1]})
        self._WrapDataVar('Switching Probability',  '', 'binomial', {'linestyle': 'b-', 'ylim': [0, 1], 'legendlabel': 'Switch. Prob.'})
        self._WrapDataVar('Detection Time', 'PreAmp Time Counts', 'normal', {'linestyle': 'r-', 'ylim': [0, PREAMP_TIMEOUT]})
        self._WrapDataVar('Detection Time Std Dev', 'PreAmp Time Counts', 'std')
        
        ###GENERAL EXPERIMENT VARIABLES######################################################################
        #####################################################################################################
        # Experiment variables that do not control any electronics explicitly can be defined here as well
        # as any data that manually entered. self._WrapExptVar('Variable Name', 'Units' [, New_Value]) method assigns
        # units and ensures that the variable was defined/set properly. It could be used to redefine the value.
        # The method returns the value of the variable.        
        reps = self._WrapExptVar('Reps')                                            # experiment repetitions
        self._WrapExptVar('Temperature', 'mK')                                      # save temperature as one extra_data experiment variable
        threshold = self._WrapExptVar('Threshold', 'PreAmp Time Counts')            # save Threshold parameter
        
        ###EXPERIMENT VARIABLES USED BY PERMANENTLY PRESENT DEVICES##########################################
        #####################################################################################################
        # Experiment variables that used by DC Rack, DAC and ADC boards should be defined here.
        
        #DC RACK TIMING VARIABLES############################################################################
        initTime = self._WrapExptVar('Init Time', 'us')                             # wait time between reps
        biasTime = self._WrapExptVar('Bias Time', 'us')                             # time of the FastBias pulse
        measTime = self._WrapExptVar('Measure Time', 'us')                          # time of the FastBias pulse
        
        #CAVITY DRIVE (READOUT) VARIABLES####################################################################
        RF1_SB_freq = self._WrapExptVar('RF1 SB Frequency', 'Hz') / G               # RF1 sideband frequency (RF1_SB_freq in GHz)
        RF1_amp = self._WrapExptVar('RF1 Amplitude', 'DAC units')                   # amplitude of the sideband modulation
        RF1_time = self._WrapExptVar('RF1 Time', 'ns')                              # length of the RF1 pulse
    
        #QUBIT DRIVE VARIABLES###############################################################################
        RF2_SB_freq = self._WrapExptVar('RF2 SB Frequency', 'Hz') / G               # RF2 sideband frequency (RF2_SB_freq in GHz)
        RF2_amp = self._WrapExptVar('RF2 Amplitude', 'DAC units')                   # amplitude of the sideband modulation
        RF2_time = self._WrapExptVar('RF2 Time', 'ns')                              # length of the RF2 pulse
  
        #JPM A VARIABLES##################################################################################### 
        JPM_bias = self._WrapExptVar('Bias Voltage', 'FastBias DAC units')          # height of the FastBias pulse   
        JPM_FPT = self._WrapExptVar('Fast Pulse Time', 'ns')                        # length of the DAC pulse
        JPM_FPA = self._WrapExptVar('Fast Pulse Amplitude', 'DAC units')            # amplitude of the DAC pulse
        JPM_FPW = self._WrapExptVar('Fast Pulse Width', 'ns')                       # DAC pulse rise time 
        
        #TIMING VARIABLES####################################################################################
        RF1toRF2 = self._WrapExptVar('RF1 to RF2 Delay', 'ns')                      # delay between the start of the RF1 pulse and the start of the RF2 pulse
        RF2toFP = self._WrapExptVar('RF2 to Fast Pulse Delay', 'ns')                # delay between the start of the RF2 pulse and the start of the fast pulse
        
        ###EXPERIMENT VARIABLES USED BY DEVICES THAT COULD BE STOLEN BY THE OTHER GRF1UP MEMBERS#############
        #####################################################################################################        
        # Experiment variables that are not be essential for some of the experiment runs should be defined here.
        # The external electronics should be called here, conditional on the presence of
        # the corresponding variables in self.Vars2Resources.
        
        #RF1 DRIVE VARIABLES#################################################################################
        if 'RF1 Attenuation' in self.Vars2Resources:                            # RF1 attenuation
            if self.Vars2Resources['RF1 Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['RF1 Attenuation']['Serial Number'], 
                                              self._WrapExptVar('RF1 Attenuation', 'dB'))

        if 'RF1 Power' in self.Vars2Resources:                                  # RF1 power
            if self.Vars2Resources['RF1 Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['RF1 Power']['GPIB Address']].Power(self._WrapExptVar('RF1 Power', 'dBm'))

        if 'RF1 Frequency' in self.Vars2Resources:                              # RF1 frequency
            if self.Vars2Resources['RF1 Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['RF1 Frequency']['GPIB Address']].Frequency(self._WrapExptVar('RF1 Frequency', 'Hz') + RF1_SB_freq * G)

        #RF2 DRIVE VARIABLES#################################################################################
        if 'RF2 Attenuation' in self.Vars2Resources:                              # RF2 attenuation
            if self.Vars2Resources['RF2 Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['RF2 Attenuation']['Serial Number'], 
                                              self._WrapExptVar('RF2 Attenuation', 'dB'))

        if 'RF2 Power' in self.Vars2Resources:                                    # RF2 power
            if self.Vars2Resources['RF2 Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['RF2 Power']['GPIB Address']].Power(self._WrapExptVar('RF2 Power', 'dBm'))

        if 'RF2 Frequency' in self.Vars2Resources:                                # RF2 frequency
            if self.Vars2Resources['RF2 Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['RF2 Frequency']['GPIB Address']].Frequency(self._WrapExptVar('RF2 Frequency', 'Hz') + RF2_SB_freq * G)
       
        ###WAVEFORMS#########################################################################################
        #####################################################################################################
        requested_waveforms = [settings['DAC A'] for settings in self.DACSettings] + [settings['DAC B'] for settings in self.DACSettings]

        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        total_time = max(max(0, -RF1toRF2) + RF1_time, max(0, RF1toRF2) + RF2_time, max(0, RF1toRF2)+ RF2toFP + JPM_smoothed_FP.size)
        
        waveforms = {};
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + total_time, 0)])
        
        if 'JPM Fast Pulse' in requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2) + RF2toFP, 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(total_time - max(0, RF1toRF2) - RF2toFP - JPM_smoothed_FP.size + DAC_ZERO_PAD_LEN, 0)])      

        if 'RF1 I' in requested_waveforms:
            if RF1_SB_freq != 0:
                waveforms['RF1 I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, -RF1toRF2), 0),
                                                pulse.CosinePulse(RF1_time, RF1_SB_freq, RF1_amp, 0, 0),
                                                pulse.DC(total_time - max(0, -RF1toRF2) - RF1_time + DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['RF1 I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, -RF1toRF2), 0),
                                                pulse.DC(RF1_time, RF1_amp),
                                                pulse.DC(total_time - max(0, -RF1toRF2) - RF1_time + DAC_ZERO_PAD_LEN, 0)])

        if 'RF1 Q' in requested_waveforms:
            if RF1_SB_freq != 0:
                waveforms['RF1 Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, -RF1toRF2), 0),
                                                pulse.SinePulse(RF1_time, RF1_SB_freq, RF1_amp, 0, 0),
                                                pulse.DC(total_time - max(0, -RF1toRF2) - RF1_time + DAC_ZERO_PAD_LEN, 0)])
            else:     
                waveforms['RF1 Q'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + total_time, 0)])

        if 'RF2 I' in requested_waveforms: 
            if RF2_SB_freq != 0:
                waveforms['RF2 I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2), 0),
                                                pulse.CosinePulse(RF2_time, RF2_SB_freq, RF2_amp, 0.0, 0.0),
                                                pulse.DC(total_time - max(0, RF1toRF2) - RF2_time + DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['RF2 I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2), 0),
                                                pulse.DC(RF2_time, RF2_amp),
                                                pulse.DC(total_time - max(0, RF1toRF2) - RF2_time + DAC_ZERO_PAD_LEN, 0)])

        if 'RF2 Q' in requested_waveforms: 
            if RF2_SB_freq != 0:
                waveforms['RF2 Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + max(0, RF1toRF2), 0),
                                                pulse.SinePulse(RF2_time, RF2_SB_freq, RF2_amp, 0.0, 0.0),
                                                pulse.DC(total_time - max(0, RF1toRF2) - RF2_time + DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['RF2 Q'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + total_time, 0)])

        for idx, settings in enumerate(self.DACSettings):
            for channel in ['DAC A', 'DAC B']:
                if self.DACSettings[idx][channel] not in waveforms:
                    raise expt.ResourceDefinitionError("'" + str(self.DACs[idx]) + "' setting '" + str(channel) + 
                        "': '" + self.DACSettings[idx][channel] + "' could not be recognized. The allowed '" + str(channel) + 
                        "' values are 'JPM Fast Pulse', 'RF1 I', 'RF1 Q', 'RF2 I', 'RF2 Q', and 'None'.")

        if PlotWaveforms:
            self._PlotWaveforms([waveforms[wf] for wf in requested_waveforms], ['r', 'g', 'b', 'k'], requested_waveforms)

        SRAMLength = len(waveforms[self.DACSettings[0]['DAC A']])
        SRAMDelay = np.ceil(SRAMLength / 1000)
        
        # Create memory command list.
        memList1 = []
        if 'FO1 FastBias Firmware Version' in self.DACSettings[0]:
            memList1.append({'Type': 'Firmware', 'Channel': 1, 'Version': self.DACSettings[0]['FO1 FastBias Firmware Version']})
        memList1.append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        memList1.append({'Type': 'Delay', 'Time': initTime})
        memList1.append({'Type': 'Bias', 'Channel': 1, 'Voltage': JPM_bias})
        memList1.append({'Type': 'Delay', 'Time': biasTime})
        memList1.append({'Type': 'SRAM', 'Start': 0, 'Length': SRAMLength, 'Delay': SRAMDelay})
        memList1.append({'Type': 'Timer', 'Time': measTime})
        memList1.append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})

        memList2 = []
        memList2.append({'Type': 'Delay', 'Time': (initTime + biasTime)})
        memList2.append({'Type': 'SRAM', 'Start': 0, 'Length': SRAMLength, 'Delay': SRAMDelay})
        memList2.append({'Type': 'Timer', 'Time': measTime})
        
        # Generate SRAM and memory for DAC boards.
        DAC1_SRAM = dac.waves2sram(waveforms[self.DACSettings[0]['DAC A']], waveforms[self.DACSettings[0]['DAC B']])
        DAC1_mem  = dac.memFromList(memList1)

        DAC2_SRAM = dac.waves2sram(waveforms[self.DACSettings[1]['DAC A']], waveforms[self.DACSettings[1]['DAC B']])
        DAC2_mem  = dac.memFromList(memList2)
        
        ###RUN###############################################################################################
        #####################################################################################################        
        P = self.LoadAndRun([DAC1_SRAM, DAC2_SRAM], [DAC1_mem, DAC2_mem], reps, 'PreAmp')
        
        ###DATA POST-PRF1CESSING##############################################################################
        #####################################################################################################
        # If the waveforms are the same but somewhat different post-processing is required then the data 
        # post-processing should be defined in a grand child of this class. Do not copy-paste the waveform 
        # specifications when it is not really necessary.
        if Histogram:
            self._PlotHistogram(P, 1)

        t_mean, t_std = self._MeanTimeFromArray(P, PREAMP_TIMEOUT)
        
        run_data = {
                    'Switching Probability': self._SwitchProbFromArray(P, threshold),
                    'Detection Time': t_mean,
                    'Detection Time Std Dev': t_std,
                   } 
        
        ###EXTRA EXPERIMENT PARAMETERS TO SAVE###############################################################
        #####################################################################################################
        self._WrapExptVar('Actual Reps', '', len(P[0]))
        return run_data, None