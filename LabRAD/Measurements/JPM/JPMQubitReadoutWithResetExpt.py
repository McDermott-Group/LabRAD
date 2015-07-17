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

class JPMQubitReadoutWithReset(expt.Experiment):
    """
    Read out a qubit connected to a resonator with a readout and a displacement (reset) pulse.
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
        RO_phase = self._WrapExptVar('Readout Phase', 'rad')                        # readout pulse phase
        Disp_amp = self._WrapExptVar('Displacement Amplitude', 'DAC units')         # amplitude of the displacement pulse
        Disp_time = self._WrapExptVar('Displacement Time', 'ns')                    # length of the displacement pulse time
        Disp_phase = self._WrapExptVar('Displacement Phase', 'rad')                 # displacement pulse phase
        
        ROtoD_offset = self._WrapExptVar('Readout to Displacement Offset', 'DAC units') # zero offset between readout and reset pulses
        #QUBIT DRIVE VARIABLES###############################################################################
        QB_SB_freq = self._WrapExptVar('Qubit SB Frequency', 'Hz') / G              # qubit sideband frequency (QB_SB_freq in GHz)
        QB_amp = self._WrapExptVar('Qubit Amplitude', 'DAC units')                  # amplitude of the sideband modulation
        QB_time = self._WrapExptVar('Qubit Time', 'ns')                             # length of the qubit pulse
      
        #JPM A VARIABLES##################################################################################### 
        JPM_bias = self._WrapExptVar('Bias Voltage', 'FastBias DAC units')          # height of the FastBias pulse   
        JPM_FPT = self._WrapExptVar('Fast Pulse Time', 'ns')                        # length of the DAC pulse
        JPM_FPA = self._WrapExptVar('Fast Pulse Amplitude', 'DAC units')            # amplitude of the DAC pulse
        JPM_FPW = self._WrapExptVar('Fast Pulse Width', 'ns')                       # DAC pulse rise time 
        
        #TIMING VARIABLES####################################################################################
        QBtoRO = self._WrapExptVar('Qubit Drive to Readout', 'ns')                  # delay between the end of the qubit pulse and the start of the readout pulse
        ROtoD = self._WrapExptVar('Readout to Displacement', 'ns')                  # delay between the end of readout pulse and the start of displacement pulse
        DtoFP = self._WrapExptVar('Displacement to Fast Pulse', 'ns')               # delay between the end of the displacement pulse and the start of the fast pulse
        
        ###EXPERIMENT VARIABLES USED BY DEVICES THAT COULD BE STOLEN BY THE OTHER GROUP MEMBERS##############
        #####################################################################################################        
        # Experiment variables that are not be essential for some of the experiment runs should be defined here.
        # The external electronics should be called here, conditional on the presence of
        # the corresponding variables in self._var2res.

        #RF VARIABLES########################################################################################
        if 'RF Attenuation' in self._var2res:                                 # RF attenuation
            if self._var2res['RF Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self._var2res['RF Attenuation']['Serial Number'], 
                                              self._WrapExptVar('RF Attenuation', 'dB'))

        if 'RF Power' in self._var2res:                                       # RF power
            if self._var2res['RF Power']['Resource'] == 'RF Generator':
                self.RFgen[self._var2res['RF Power']['GPIB Address']].Power(self._WrapExptVar('RF Power', 'dBm'))

        if 'RF Frequency' in self._var2res:                               # RF frequency
            if self._var2res['RF Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self._var2res['RF Frequency']['GPIB Address']].Frequency(self._WrapExptVar('RF Frequency', 'Hz'))

        #RF DRIVE (READOUT) VARIABLES########################################################################
        if 'Readout Attenuation' in self._var2res:                            # readout attenuation
            if self._var2res['Readout Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self._var2res['Readout Attenuation']['Serial Number'], 
                                              self._WrapExptVar('Readout Attenuation', 'dB'))

        if 'Readout Power' in self._var2res:                                  # readout power
            if self._var2res['Readout Power']['Resource'] == 'RF Generator':
                self.RFgen[self._var2res['Readout Power']['GPIB Address']].Power(self._WrapExptVar('Readout Power', 'dBm'))

        if 'Readout Frequency' in self._var2res:                              # readout frequency
            if self._var2res['Readout Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self._var2res['Readout Frequency']['GPIB Address']].Frequency(self._WrapExptVar('Readout Frequency', 'Hz') + RO_SB_freq * G)

        #QUBIT VARIABLES#####################################################################################
        if 'Qubit Attenuation' in self._var2res:                              # qubit attenuation
            if self._var2res['Qubit Attenuation']['Resource'] == 'Lab Brick Attenuator':
            
                self.cxn(self._var2res['Qubit Attenuation']['Serial Number'], 
                                              self._WrapExptVar('Qubit Attenuation', 'dB'))

        if 'Qubit Power' in self._var2res:                                    # qubit power
            if self._var2res['Qubit Power']['Resource'] == 'RF Generator':
                self.RFgen[self._var2res['Qubit Power']['GPIB Address']].Power(self._WrapExptVar('Qubit Power', 'dBm'))

        if 'Qubit Frequency' in self._var2res:                                # qubit frequency
            if self._var2res['Qubit Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self._var2res['Qubit Frequency']['GPIB Address']].Frequency(self._WrapExptVar('Qubit Frequency', 'Hz') + QB_SB_freq * G)
                
        #DC BIAS VARIABLES###################################################################################
        if 'Qubit Flux Bias Voltage' in self.ExptVars and 'Qubit Flux Bias Voltage' in self._var2res:     # qubit flux bias
            if self._var2res['Qubit Flux Bias Voltage']['Resource'] == 'SIM':
                self.SIM[(self._var2res['Qubit Flux Bias Voltage']['GPIB Address'], 
                          self._var2res['Qubit Flux Bias Voltage']['SIM Slot'])].Voltage(self._WrapExptVar('Qubit Flux Bias Voltage', 'V'))
        
        ###WAVEFORMS#########################################################################################
        #####################################################################################################
        requested_waveforms = [settings['DAC A'] for settings in self.DACSettings] + [settings['DAC B'] for settings in self.DACSettings]

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
            if QB_SB_freq != 0:
                waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                  pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                                  pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])
            else:
                waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                  pulse.DC(QB_time, QB_amp),
                                                  pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Qubit Q' in requested_waveforms: 
            if QB_SB_freq != 0:
                waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                  pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                                  pulse.DC(QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])
            else:
                waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time + ROtoD + Disp_time + FPtoEnd, 0)])

        if 'Readout I' in requested_waveforms:
            if RO_SB_freq != 0:
                waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase / (2 * np.pi), 0),
                                                    pulse.CosinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase / (2 * np.pi), 0),
                                                    pulse.CosinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase / (2 * np.pi), 0),
                                                    pulse.DC(FPtoEnd, 0)])
            else:
                waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.DC(RO_time, RO_amp * np.cos(RO_phase)),
                                                    pulse.DC(ROtoD, ROtoD_offset),
                                                    pulse.DC(Disp_time, Disp_amp * np.cos(Disp_phase)),
                                                    pulse.DC(FPtoEnd, 0)])

        if 'Readout Q' in requested_waveforms:
            if RO_SB_freq != 0:
                waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, RO_phase / (2 * np.pi), 0),
                                                    pulse.SinePulse(ROtoD, RO_SB_freq, ROtoD_offset, ROtoD * RO_SB_freq + RO_phase / (2 * np.pi), 0),
                                                    pulse.SinePulse(Disp_time, RO_SB_freq, Disp_amp, (RO_time + ROtoD) * RO_SB_freq + Disp_phase / (2 * np.pi), 0),
                                                    pulse.DC(FPtoEnd, 0)])
            else:     
                waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.DC(RO_time, RO_amp * np.sin(RO_phase)),
                                                    pulse.DC(ROtoD, ROtoD_offset),
                                                    pulse.DC(Disp_time, Disp_amp * np.sin(Disp_phase)),
                                                    pulse.DC(FPtoEnd, 0)])

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