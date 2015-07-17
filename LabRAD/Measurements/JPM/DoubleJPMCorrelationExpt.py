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

class DoubleJPMCorrelation(expt.Experiment):
    """
    Read out two JPMs in a correlation experiment.
    """
    def RunOnce(self, Histogram=False, PlotWaveforms=False):
        ###DATA VARIABLES####################################################################################
        #####################################################################################################
        # Units for data variables as well as plotting preferences can be defined here.
        # Example: self._WrapDataVar('P',  '', 'binomial', {'name': 'Probability', 'linestyle': 'b-', 'linewidth': 2, 'legendlabel': 'Prob.', 'ylim': [0, 1]})
        self._WrapDataVar('Pa',  '', 'binomial', {'linestyle': 'r-', 'ylim': [0, 1]})
        self._WrapDataVar('Pb',  '', 'binomial', {'linestyle': 'b-', 'ylim': [0, 1]})
        self._WrapDataVar('P00', '', 'binomial', {'linestyle': 'k-', 'ylim': [0, 1]})
        self._WrapDataVar('P01', '', 'binomial', {'linestyle': 'g-', 'ylim': [0, 1]})
        self._WrapDataVar('P10', '', 'binomial', {'linestyle': 'c-', 'ylim': [0, 1]})
        self._WrapDataVar('P11', '', 'binomial', {'linestyle': 'm-', 'ylim': [0, 1]})
        self._WrapDataVar('JPM A Detection Time', 'PreAmp Time Counts', 'normal', {'linestyle': 'r-', 'ylim': [0, PREAMP_TIMEOUT]})
        self._WrapDataVar('JPM B Detection Time', 'PreAmp Time Counts', 'normal', {'linestyle': 'b-', 'ylim': [0, PREAMP_TIMEOUT]})
        self._WrapDataVar('Detection Time Diff', 'PreAmp Time Counts', 'normal', {'linestyle': 'k-', 'ylim': [-PREAMP_TIMEOUT, PREAMP_TIMEOUT]})
        self._WrapDataVar('JPM A Detection Time Std Dev', 'PreAmp Time Counts', 'std')
        self._WrapDataVar('JPM B Detection Time Std Dev', 'PreAmp Time Counts', 'std')
        self._WrapDataVar('Detection Time Diff Std Dev', 'PreAmp Time Counts', 'std')
 
        ###GENERAL EXPERIMENT VARIABLES######################################################################
        #####################################################################################################
        # Experiment variables that do not control any electronics explicitly can be defined here as well
        # as any data that manually entered. self._WrapExptVar('Variable Name', 'Units' [, New_Value]) method assigns
        # units and ensures that the variable was defined/set properly. It could be used to redefine the value.
        # The method returns the value of the variable.        
        reps = self._WrapExptVar('Reps')                                                  # experiment repetitions
        self._WrapExptVar('Temperature', 'mK')                                            # save temperature as one extra experiment variable
        threshold = self._WrapExptVar('Threshold', 'PreAmp Time Counts')                  # save Threshold parameter
        
        ###EXPERIMENT VARIABLES USED BY PERMANENTLY PRESENT DEVICES##########################################
        #####################################################################################################
        # Experiment variables that used by DAC and ADC cards should be defined here.
        
        #DC RACK TIMING VARIABLES############################################################################
        initTime = self._WrapExptVar('Init Time', 'us')                                   # wait time between reps
        biasTime = self._WrapExptVar('Bias Time', 'us')                                   # time of the FastBias pulse
        measTime = self._WrapExptVar('Measure Time', 'us')                                # time of the FastBias pulse

        #RF DRIVE VARIABLES##################################################################################
        RF_SBfreq = self._WrapExptVar('RF Sideband Frequency', 'Hz') / G                  # sideband frequency for sideband modulation (RF_SBfreq in GHz)
        RF_time = self._WrapExptVar('RF Time', 'ns')                                      # length of the RF pulse
        RF_amp = self._WrapExptVar('RF Amplitude', 'DAC units')                           # amplitude of the sideband modulation
        
        #JPM A VARIABLES##################################################################################### 
        JPMA_bias = self._WrapExptVar('JPM A Bias Voltage', 'FastBias DAC units')         # height of the FastBias pulse   
        JPMA_FPT = self._WrapExptVar('JPM A Fast Pulse Time', 'ns')                       # length of the DAC pulse
        JPMA_FPA = self._WrapExptVar('JPM A Fast Pulse Amplitude', 'DAC units')           # amplitude of the DAC pulse
        JPMA_FW = self._WrapExptVar('JPM A Fast Pulse Width', 'ns')                       # DAC pulse rise time 
        
        #JPM B VARIABLES#####################################################################################
        JPMB_bias = self._WrapExptVar('JPM B Bias Voltage', 'FastBias DAC units')         # height of the FastBias pulse   
        JPMB_FPT = self._WrapExptVar('JPM B Fast Pulse Time', 'ns')                       # length of the DAC pulse
        JPMB_FPA = self._WrapExptVar('JPM B Fast Pulse Amplitude', 'DAC units')           # amplitude of the DAC pulse
        JPMB_FW = self._WrapExptVar('JPM B Fast Pulse Width', 'ns')                       # DAC pulse rise time
        
        #TIMING VARIABLES####################################################################################
        RFtoFP_A = self._WrapExptVar('RF to JPM A Fast Pulse Delay', 'ns')                # delay between the start of the RF pulse and the start of the JPM A pulse
        JPMdelay = self._WrapExptVar('JPM A to JPM B Fast Pulse Delay', 'ns')             # delay between the JPM A and JPM B fast pulses (can be negative, i.e. the JPM B pulse is ahead of the JPM A one)
        
        ###EXPERIMENT VARIABLES USED BY DEVICES THAT COULD BE STOLEN BY THE OTHER GROUP MEMBERS##############
        #####################################################################################################        
        # Experiment variables that are not be essential for some of the experiment runs should be defined here.
        # The external electronics should be called here, conditional on the presence of
        # the corresponding variables in self.Vars2Resources.
        
        #RF DRIVE VARIABLES##################################################################################
        if 'RF Attenuation' in self.Vars2Resources:                                      # RF attenuation
            if self.Vars2Resources['RF Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['RF Attenuation']['Serial Number'], 
                                              self._WrapExptVar('RF Attenuation', 'dB'))

        if 'RF Power' in self.Vars2Resources:                                            # RF power
            if self.Vars2Resources['RF Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['RF Power']['GPIB Address']].Power(self._WrapExptVar('RF Power', 'dBm'))

        if 'RF Frequency' in self.Vars2Resources:                                        # RF frequency
            if self.Vars2Resources['RF Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['RF Frequency']['GPIB Address']].Frequency(self._WrapExptVar('RF Frequency', 'Hz') + RF_SBfreq * G)
        #DC BIAS VARIABLES###################################################################################
        if 'DC Bias' in self.ExptVars and 'DC Bias' in self.Vars2Resources:              # DC bias
            if self.Vars2Resources['DC Bias']['Resource'] == 'SIM':
                self.SIM[(self.Vars2Resources['DC Bias']['GPIB Address'], 
                          self.Vars2Resources['DC Bias']['SIM Slot'])].Voltage(self._WrapExptVar('DC Bias', 'V'))
        
        ###WAVEFORMS#########################################################################################
        #####################################################################################################
        requested_waveforms = [settings['DAC A'] for settings in self.DACSettings] + [settings['DAC B'] for settings in self.DACSettings]

        JPMA_FP = pulse.GaussPulse(JPMA_FPT, JPMA_FW, JPMA_FPA)
        JPMB_FP = pulse.GaussPulse(JPMB_FPT, JPMB_FW, JPMB_FPA)
        
        waveforms = {};
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + RF_time, 0)])
        
        # Set up fast pulses.
        if 'JPM A Fast Pulse' in requested_waveforms:
            waveforms['JPM A Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + RFtoFP_A, 0),
                                                      JPMA_FP,
                                                      pulse.DC(RF_time - RFtoFP_A - JPMA_FP.size + DAC_ZERO_PAD_LEN, 0)])

        if 'JPM B Fast Pulse' in requested_waveforms:
            waveforms['JPM B Fast Pulse'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + RFtoFP_A + JPMdelay, 0),
                                                      JPMB_FP,
                                                      pulse.DC(RF_time - RFtoFP_A - JPMdelay - JPMB_FP.size + DAC_ZERO_PAD_LEN, 0)])
        # Set up RF pulse.
        if 'RF I' in requested_waveforms:
            if RF_SBfreq != 0:        
                waveforms['RF I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                               pulse.CosinePulse(RF_time, RF_SBfreq, RF_amp, 0.0, 0.0),
                                               pulse.DC(DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['RF I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                               pulse.DC(RF_time, RF_amp),
                                               pulse.DC(DAC_ZERO_PAD_LEN, 0)])  

        if 'RF Q' in requested_waveforms:
            if RF_SBfreq != 0:        
                waveforms['RF Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                               pulse.SinePulse(RF_time, RF_SBfreq, RF_amp, 0.0, 0.0),
                                               pulse.DC(DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['RF Q'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + RF_time, 0)])

        for idx, settings in enumerate(self.DACSettings):
            for channel in ['DAC A', 'DAC B']:
                if self.DACSettings[idx][channel] not in waveforms:
                    raise expt.ResourceDefinitionError("'" + str(self.DACs[idx]) + "' setting '" + str(channel) + 
                        "': '" + self.DACSettings[idx][channel] + "' could not be recognized. The allowed '" + str(channel) + 
                        "' values are 'JPM A Fast Pulse', 'JPM B Fast Pulse', 'RF I', 'RF Q', and 'None'.")

        if PlotWaveforms:
            self._PlotWaveforms([waveforms[wf] for wf in requested_waveforms], ['r', 'g', 'b', 'k'], requested_waveforms)

        SRAMLength = len(waveforms[self.DACSettings[0]['DAC A']])
        SRAMDelay = np.ceil(SRAMLength / 1000)
        
        # Create memory command list.
        memList1 = []
        if 'FO1 FastBias Firmware Version' in self.DACSettings[0]:
            memList1.append({'Type': 'Firmware', 'Channel': 1, 'Version': self.DACSettings[0]['FO1 FastBias Firmware Version']})
        if 'FO2 FastBias Firmware Version' in self.DACSettings[0]:
            memList1.append({'Type': 'Firmware', 'Channel': 2, 'Version': self.DACSettings[0]['FO2 FastBias Firmware Version']})
        memList1.append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        memList1.append({'Type': 'Bias', 'Channel': 2, 'Voltage': 0})
        memList1.append({'Type': 'Delay', 'Time': initTime})
        memList1.append({'Type': 'Bias', 'Channel': 1, 'Voltage': JPMA_bias})
        memList1.append({'Type': 'Bias', 'Channel': 2, 'Voltage': JPMB_bias})
        memList1.append({'Type': 'Delay', 'Time': biasTime})
        memList1.append({'Type': 'SRAM', 'Start': 0, 'Length': SRAMLength, 'Delay': SRAMDelay})
        memList1.append({'Type': 'Timer', 'Time': measTime})
        memList1.append({'Type': 'Bias', 'Channel': 1, 'Voltage': 0})
        memList1.append({'Type': 'Bias', 'Channel': 2, 'Voltage': 0})
        
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
        P = self.LoadAndRun([DAC1_SRAM, DAC2_SRAM], [DAC1_mem, DAC2_mem], reps, 'MultiPreAmp')
        
        ###DATA POST-PROCESSING##############################################################################
        #####################################################################################################
        # If the waveforms are the same but somewhat different post-processing is required then the data 
        # post-processing should be defined in a grand child of this class. Do not copy-paste the waveform 
        # specifications when it is not really necessary.
        if Histogram:
            self._PlotHistogram(P, 2)

        ta_mean, ta_std = self._MeanTimeFromArray(P[0], PREAMP_TIMEOUT)
        tb_mean, tb_std = self._MeanTimeFromArray(P[1], PREAMP_TIMEOUT)
        dt_mean, dt_std = self._MeanTimeDiffFromArray(P, PREAMP_TIMEOUT)
        
        p = self._PreAmpTimeCountsTo10Array(P, threshold)
        
        pa = p[0, :]
        pb = p[1, :]
        n = float(len(pa))
        run_data = {
                    'Pa':  float(pa.sum()) / n,
                    'Pb':  float(pb.sum()) / n,
                    'P11': float((pa * pb).sum()) / n,
                    'P00': float(((1 - pa) * (1 - pb)).sum()) / n,
                    'P10': float((pa * (1 - pb)).sum()) / n,
                    'P01': float(((1 - pa) * pb).sum()) / n,
                    'JPM A Detection Time': ta_mean,
                    'JPM B Detection Time': tb_mean,
                    'Detection Time Diff': dt_mean,
                    'JPM A Detection Time Std Dev': ta_std,
                    'JPM B Detection Time Std Dev': tb_std,
                    'Detection Time Diff Std Dev': dt_std
                   } 
        
        ###EXTRA EXPERIMENT PARAMETERS TO SAVE###############################################################
        #####################################################################################################
        self._WrapExptVar('Actual Reps', '', n)
        return run_data, None