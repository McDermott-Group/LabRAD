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

class JPMStarkShift(expt.Experiment):
    """
    Stark shift experiment with a JPM.
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
        RO_SB_freq = self._WrapExptVar('Readout SB Frequency', 'Hz') / G            # readout sideband frequency (RO_SB_freq in GHz)
        RO_amp = self._WrapExptVar('Readout Amplitude', 'DAC units')                # amplitude of the sideband modulation
        RO_time = self._WrapExptVar('Readout Time', 'ns')                           # length of the readout pulse
        
        #CAVITY DRIVE (STARK SHIFT) VARIABLES################################################################
        Stark_amp = self._WrapExptVar('Stark Amplitude', 'DAC units')               # amplitude of the sideband modulation
        Stark_time = self._WrapExptVar('Stark Time', 'ns')                          # length of the Stark pulse
        
        #QUBIT DRIVE VARIABLES###############################################################################
        QB_SB_freq = self._WrapExptVar('Qubit SB Frequency', 'Hz') / G              # qubit sideband frequency (RO_SB_freq in GHz)
        QB_amp = self._WrapExptVar('Qubit Amplitude', 'DAC units')                  # amplitude of the sideband modulation
        QB_time = self._WrapExptVar('Qubit Time', 'ns')                             # length of the qubit pulse
      
        #JPM A VARIABLES##################################################################################### 
        JPM_bias = self._WrapExptVar('Bias Voltage', 'FastBias DAC units')          # height of the FastBias pulse   
        JPM_FPT = self._WrapExptVar('Fast Pulse Time', 'ns')                        # length of the DAC pulse
        JPM_FPA = self._WrapExptVar('Fast Pulse Amplitude', 'DAC units')            # amplitude of the DAC pulse
        JPM_FPW = self._WrapExptVar('Fast Pulse Width', 'ns')                       # DAC pulse rise time 
        
        #TIMING VARIABLES####################################################################################
        QBtoRO = self._WrapExptVar('Qubit Drive to Readout Delay', 'ns')            # delay from the start of the qubit pulse to the start of the readout pulse
        ROtoFP = self._WrapExptVar('Readout to Fast Pulse Delay', 'ns')             # delay from the start of the RF pulse to the start of the fast pulse
        
        ###EXPERIMENT VARIABLES USED BY DEVICES THAT COULD BE STOLEN BY THE OTHER GROUP MEMBERS##############
        #####################################################################################################        
        # Experiment variables that are not be essential for some of the experiment runs should be defined here.
        # The external electronics should be called here, conditional on the presence of
        # the corresponding variables in self.Vars2Resources.
        
        #RF DRIVE (READOUT) VARIABLES########################################################################
        if 'Readout Attenuation' in self.Vars2Resources:                            # readout attenuation
            if self.Vars2Resources['Readout Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['Readout Attenuation']['Serial Number'], 
                                              self._WrapExptVar('Readout Attenuation', 'dB'))

        if 'Readout Power' in self.Vars2Resources:                                  # readout power
            if self.Vars2Resources['Readout Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Readout Power']['GPIB Address']].Power(self._WrapExptVar('Readout Power', 'dBm'))

        if 'Readout Frequency' in self.Vars2Resources:                              # readout frequency
            if self.Vars2Resources['Readout Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Readout Frequency']['GPIB Address']].Frequency(self._WrapExptVar('Readout Frequency', 'Hz') + RO_SB_freq * G)

        #QUBIT VARIABLES#####################################################################################
        if 'Qubit Attenuation' in self.Vars2Resources:                              # qubit attenuation
            if self.Vars2Resources['Qubit Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['Qubit Attenuation']['Serial Number'], 
                                              self._WrapExptVar('Qubit Attenuation', 'dB'))

        if 'Qubit Power' in self.Vars2Resources:                                    # qubit power
            if self.Vars2Resources['Qubit Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Qubit Power']['GPIB Address']].Power(self._WrapExptVar('Qubit Power', 'dBm'))

        if 'Qubit Frequency' in self.Vars2Resources:                                # qubit frequency
            if self.Vars2Resources['Qubit Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Qubit Frequency']['GPIB Address']].Frequency(self._WrapExptVar('Qubit Frequency', 'Hz') + QB_SB_freq * G)
                
        #DC BIAS VARIABLES###################################################################################
        if 'Qubit Flux Bias Voltage' in self.ExptVars and 'Qubit Flux Bias Voltage' in self.Vars2Resources:     # qubit flux bias
            if self.Vars2Resources['Qubit Flux Bias Voltage']['Resource'] == 'SIM':
                self.SIM[(self.Vars2Resources['Qubit Flux Bias Voltage']['GPIB Address'], 
                          self.Vars2Resources['Qubit Flux Bias Voltage']['SIM Slot'])].Voltage(self._WrapExptVar('Qubit Flux Bias Voltage', 'V'))
        
        ###WAVEFORMS#########################################################################################
        #####################################################################################################
        requested_waveforms = [settings['DAC A'] for settings in self.DACSettings] + [settings['DAC B'] for settings in self.DACSettings]

        JPM_smoothed_FP = pulse.GaussPulse(JPM_FPT, JPM_FPW, JPM_FPA)
        QBtoEnd = QBtoRO + max(0, -RO_time - ROtoFP) + RO_time + max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN
        
        waveforms = {};
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoEnd, 0)])

        if 'JPM Fast Pulse' in requested_waveforms:
            waveforms['JPM Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoRO + max(0, RO_time + ROtoFP), 0),
                                                     JPM_smoothed_FP,
                                                     pulse.DC(max(0, -JPM_smoothed_FP.size - ROtoFP) + DAC_ZERO_PAD_LEN, 0)])

        if 'Readout I' in requested_waveforms:
            if RO_SB_freq != 0:
                waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                    pulse.CosinePulse(Stark_time, RO_SB_freq, Stark_amp, 0.0, 0.0),
                                                    pulse.DC(QBtoRO + max(0, -RO_time - ROtoFP), 0),
                                                    pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                    pulse.DC(max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                    pulse.DC(Stark_time, Stark_amp),
                                                    pulse.DC(QBtoRO + max(0, -RO_time - ROtoFP), 0),
                                                    pulse.DC(RO_time, RO_amp),
                                                    pulse.DC(max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in requested_waveforms:
            if RO_SB_freq != 0:
                waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                    pulse.SinePulse(Stark_time, RO_SB_freq, Stark_amp, 0.0, 0.0),
                                                    pulse.DC(QBtoRO + max(0, -RO_time - ROtoFP), 0),
                                                    pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                    pulse.DC(max(0, ROtoFP + JPM_smoothed_FP.size) + DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoEnd, 0)])

        if 'Qubit I' in requested_waveforms:
            if QB_SB_freq != 0:
                waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                                  pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                                  pulse.DC(QBtoEnd, 0)])
            else:
                waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                                  pulse.DC(QB_time, QB_amp),
                                                  pulse.DC(QBtoEnd, 0)])

        if 'Qubit Q' in requested_waveforms:
            if QB_SB_freq != 0:                                                  
                waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time - QB_time, 0),
                                                  pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                                  pulse.DC(QBtoEnd, 0)])
            else:
                waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + Stark_time + QBtoEnd, 0)])

        for idx, settings in enumerate(self.DACSettings):
            for channel in ['DAC A', 'DAC B']:
                if self.DACSettings[idx][channel] not in waveforms:
                    raise expt.ResourceDefinitionError("'" + str(self.DACs[idx]) + "' setting '" + str(channel) + 
                        "': '" + self.DACSettings[idx][channel] + "' could not be recognized. The allowed '" + str(channel) + 
                        "' values are 'JPM Fast Pulse', 'Readout I', 'Readout Q', 'Qubit I', 'Qubit Q', and 'None'.")

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
        ###DATA POST-PROCESSING##############################################################################
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