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
import LabRAD.Servers.Instruments.GHzBoards.ghz_fpga_control as dac
import LabRAD.Servers.Instruments.GHzBoards.pulse_shapes as pulse

DAC_ZERO_PAD_LEN = 20
PREAMP_TIMEOUT = 1253

class JPMQubitReadoutWithReset(expt.Experiment):
    """
    Read out a qubit connected to a resonator with a readout and a displacement (reset) pulse.
    """
    def run_once(self, Histogram=False, PlotWaveforms=False):
        #RF VARIABLES##############################################################################
        self.send_request('RF Attenuation', False)             # RF attenuation
        self.send_request('RF Power', False)                   # RF power
        self.send_request('RF Frequency', False,               # RF frequency
                self.variable('RF Frequency') + self.variable('RF SB Frequency'))

        #QUBIT VARIABLES###########################################################################
        self.send_request('Qubit Attenuation', False)          # qubit attenuation
        self.send_request('Qubit Power', False)                # qubit power
        self.send_request('Qubit Frequency', False,            # qubit frequency
                self.variable('Qubit Frequency') + self.variable('Qubit SB Frequency'))
    
        #RF DRIVE (READOUT) VARIABLES##############################################################
        self.send_request('Readout Attenuation', False)        # readout attenuation
        self.send_request('Readout Power', False)              # readout power
        self.send_request('Readout Frequency', False,          # readout frequency
            self.variable('Readout Frequency') + self.variable('Readout SB Frequency'))

        #DC BIAS VARIABLES#########################################################################
        self.send_request('Qubit Flux Bias Voltage', False)    # qubit flux bias
        
        ###EXPERIMENT VARIABLES USED BY PERMANENTLY PRESENT DEVICES################################
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
        
  
        ###RUN#####################################################################################
        self.acknowledge_requests()
        P = self.LoadAndRun([DAC1_SRAM, DAC2_SRAM], [DAC1_mem, DAC2_mem], reps, 'PreAmp')
        
        ###DATA POST-PROCESSING##############################################################################
        #####################################################################################################
        # If the waveforms are the same but somewhat different post-processing is required then the data 
        # post-processing should be defined in a grand child of this class. Do not copy-paste the waveform 
        # specifications when it is not really necessary.
        if Histogram:
            self._PlotHistogram(P, 1)

        t_mean, t_std = self._MeanTimeFromArray(P, PREAMP_TIMEOUT)
        
        ###DATA STRUCTURE##########################################################################
        run_data = {
                    'Switching Probability': {'Value': self._SwitchProbFromArray(P, threshold),
                                              'Distribution': 'binomial',
                                              'Preferances':  {'linestyle': 'b-',
                                                               'ylim': [0, 1],
                                                               'legendlabel': 'Switch. Prob.'}},
                    'Detection Time': {'Value': t_mean * units.PreAmpTimeCounts,
                                       'Distribution': 'normal',
                                       'Preferances': {'linestyle': 'r-', 
                                                       'ylim': [0, PREAMP_TIMEOUT]}},
                    'Detection Time Std Dev': t_std * units.PreAmpTimeCounts,
                   } 
        
        ###EXTRA EXPERIMENT PARAMETERS TO SAVE###############################################################
        #####################################################################################################
        self.add_var('Actual Reps', len(P[0]))
        
        return run_data, None