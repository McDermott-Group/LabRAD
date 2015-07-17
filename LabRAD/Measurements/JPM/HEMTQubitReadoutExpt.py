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

class HEMTQubitReadout(expt.Experiment):
    """
    Read out a qubit connected to a resonator.
    """
    def RunOnce(self, ADCName=None, PlotWaveforms=False):
        ###DATA VARIABLES####################################################################################
        #####################################################################################################
        # Units for data variables as well as plotting preferences can be defined here.
        # Example: self._WrapDataVar('P',  '', 'binomial', ' {'name': 'Probability', 'linestyle': 'b-', 'linewidth': 2, 'legendlabel': 'Prob.', 'ylim': [0, 1]})
        self._WrapDataVar('I', 'ADC units', None, {'linestyle': 'b-', 'linewidth': 1})
        self._WrapDataVar('Q', 'ADC units', None, {'linestyle': 'g-', 'linewidth': 1})
        
        ###GENERAL EXPERIMENT VARIABLES######################################################################
        #####################################################################################################
        # Experiment variables that do not control any electronics explicitly can be defined here as well
        # as any data that manually entered. self._WrapExptVar('Variable Name', 'Units' [, New_Value]) method assigns
        # units and ensures that the variable was defined/set properly. It could be used to redefine the value.
        # The method returns the value of the variable.        
        reps = self._WrapExptVar('Reps')                                    # experiment repetitions
        self._WrapExptVar('Temperature', 'mK')                              # save temperature as one extra_data experiment variable
        
        ###EXPERIMENT VARIABLES USED BY PERMANENTLY PRESENT DEVICES##########################################
        #####################################################################################################
        # Experiment variables that used by DC Rack, DAC and ADC boards should be defined here.
        
        #DC RACK TIMING VARIABLES############################################################################
        initTime = self._WrapExptVar('Init Time', 'us')                     # wait time between reps
        
        #CAVITY DRIVE (READOUT) VARIABLES####################################################################
        RO_SB_freq = self._WrapExptVar('Readout SB Frequency', 'Hz') / G    # readout sideband frequency (RO_SB_freq in GHz)
        RO_amp = self._WrapExptVar('Readout Amplitude', 'DAC units')        # amplitude of the sideband modulation
        RO_time = self._WrapExptVar('Readout Time', 'ns')                   # length of the readout pulse
        
        #QUBIT DRIVE VARIABLES###############################################################################
        QB_SB_freq = self._WrapExptVar('Qubit SB Frequency', 'Hz') / G      # qubit sideband frequency (RO_SB_freq in GHz)
        QB_amp = self._WrapExptVar('Qubit Amplitude', 'DAC units')          # amplitude of the sideband modulation
        QB_time = self._WrapExptVar('Qubit Time', 'ns')                     # length of the qubit pulse
      
        #TIMING VARIABLES####################################################################################
        QBtoRO = self._WrapExptVar('Qubit Drive to Readout Delay', 'ns')    # delay from the start of the qubit pulse to the start of the readout pulse
        ADC_wait_time = self._WrapExptVar('ADC Wait Time', 'ns')            # delay from the start of the readout pulse to the start of the demodulation
        
        ###EXPERIMENT VARIABLES USED BY DEVICES THAT COULD BE STOLEN BY THE OTHER GROUP MEMBERS##############
        #####################################################################################################        
        # Experiment variables that are not be essential for some of the experiment runs should be defined here.
        # The external electronics should be called here, conditional on the presence of
        # the corresponding variables in self.Vars2Resources.
        
        #RF DRIVE (READOUT) VARIABLES########################################################################
        if 'Readout Attenuation' in self.Vars2Resources:                    # readout attenuation
            if self.Vars2Resources['Readout Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['Readout Attenuation']['Serial Number'], 
                                              self._WrapExptVar('Readout Attenuation', 'dB'))

        if 'Readout Power' in self.Vars2Resources:                          # readout power
            if self.Vars2Resources['Readout Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Readout Power']['GPIB Address']].Power(self._WrapExptVar('Readout Power', 'dBm'))

        if 'Readout Frequency' in self.Vars2Resources:                      # readout frequency
            if self.Vars2Resources['Readout Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Readout Frequency']['GPIB Address']].Frequency(self._WrapExptVar('Readout Frequency', 'Hz') + RO_SB_freq * G)

        #QUBIT VARIABLES#####################################################################################
        if 'Qubit Attenuation' in self.Vars2Resources:                      # qubit attenuation
            if self.Vars2Resources['Qubit Attenuation']['Resource'] == 'Lab Brick':
                self.LabBricks.SetAttenuation(self.Vars2Resources['Qubit Attenuation']['Serial Number'], 
                                              self._WrapExptVar('Qubit Attenuation', 'dB'))

        if 'Qubit Power' in self.Vars2Resources:                            # qubit power
            if self.Vars2Resources['Qubit Power']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Qubit Power']['GPIB Address']].Power(self._WrapExptVar('Qubit Power', 'dBm'))

        if 'Qubit Frequency' in self.Vars2Resources:                        # qubit frequency
            if self.Vars2Resources['Qubit Frequency']['Resource'] == 'RF Generator':
                self.RFgen[self.Vars2Resources['Qubit Frequency']['GPIB Address']].Frequency(self._WrapExptVar('Qubit Frequency', 'Hz') + QB_SB_freq * G)
                
        #DC BIAS VARIABLES###################################################################################
        if 'Flux Bias Voltage' in self.ExptVars and 'Flux Bias Voltage' in self.Vars2Resources:     # flux bias
            if self.Vars2Resources['Flux Bias Voltage']['Resource'] == 'SIM':
                self.SIM[(self.Vars2Resources['Flux Bias Voltage']['GPIB Address'], 
                          self.Vars2Resources['Flux Bias Voltage']['SIM Slot'])].Voltage(self._WrapExptVar('Flux Bias Voltage', 'V'))
        
        ###WAVEFORMS#########################################################################################
        #####################################################################################################
        requested_waveforms = [settings['DAC A'] for settings in self.DACSettings] + [settings['DAC B'] for settings in self.DACSettings]

        waveforms = {};
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])
        
        if 'Readout I' in requested_waveforms:
            if RO_SB_freq != 0:
                waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                    pulse.DC(DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.DC(RO_time, RO_amp),
                                                    pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in requested_waveforms:
            if RO_SB_freq != 0:
                waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                    pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                    pulse.DC(DAC_ZERO_PAD_LEN, 0)])
            else:
                waveforms['Readout Q'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])

        if 'Qubit I' in requested_waveforms:       
            if QB_SB_freq != 0:        
                waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                  pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                                  pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])
            else:    
                waveforms['Qubit I'] =  np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                   pulse.DC(QB_time, QB_amp),
                                                   pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])

        if 'Qubit Q' in requested_waveforms:        
            if QB_SB_freq != 0:        
                waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                                  pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                                  pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])
            else
                waveforms['Qubit Q'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])
 
        for idx, settings in enumerate(self.DACSettings):
            for channel in ['DAC A', 'DAC B']:
                if self.DACSettings[idx][channel] not in waveforms:
                    raise expt.ResourceDefinitionError("'" + str(self.DACs[idx]) + "' setting '" + str(channel) + 
                        "': '" + self.DACSettings[idx][channel] + "' could not be recognized. The allowed '" + str(channel) + 
                        "' values are 'Readout I', 'Readout Q', 'Qubit I', 'Qubit Q', and 'None'.")

        if PlotWaveforms:
            self._PlotWaveforms([waveforms[wf] for wf in requested_waveforms], ['r', 'g', 'b', 'k'], requested_waveforms)

        SRAMLength = len(waveforms[self.DACSettings[0]['DAC A']])
        SRAMDelay = np.ceil(SRAMLength / 1000)
                              
        ADCName = self._GetADCName(ADCName)
        self.ADCSettings[self.ADCs.index(ADCName)]['ADCDelay'] = DAC_ZERO_PAD_LEN + ADC_wait_time + QB_time + QBtoRO        # Waiting time before starting demodulation.
        self.ADCSettings[self.ADCs.index(ADCName)]['DemodFreq'] = -RO_SB_freq * G

        DAC1_SRAM = dac.waves2sram(waveforms[self.DACSettings[0]['DAC A']], waveforms[self.DACSettings[0]['DAC B']])
        DAC2_SRAM = dac.waves2sram(waveforms[self.DACSettings[1]['DAC A']], waveforms[self.DACSettings[1]['DAC B']])
        DAC_mem =  dac.memSimple(initTime, SRAMLength, 0, SRAMDelay)
        
        ###RUN###############################################################################################
        #####################################################################################################
        result = self.LoadAndRun([DAC1_SRAM, DAC2_SRAM], [DAC_mem, DAC_mem], reps, 'ADC')
        
        ###DATA POST-PROCESSING##############################################################################
        #####################################################################################################
        # If the waveforms are the same but somewhat different post-processing is required then the data 
        # post-processing should be defined in a grand child of this class. Do not copy-paste the waveform 
        # specifications when it is not really necessary.
        Is, Qs = result[0] 
        
        run_data = {'I': np.array(Is),
                    'Q': np.array(Qs)}
        
        extra_data = {}
        if self.ADCSettings[self.ADCs.index(ADCName)]['RunMode'] == 'demodulate':
            self._WrapDataVar('Single Shot Is', 'ADC units', None, {'linestyle': 'b.'})
            self._WrapDataVar('Single Shot Qs', 'ADC units', None, {'linestyle': 'g.'})
            self._WrapDataVar('I', 'ADC units', 'normal', {'linestyle': 'b-'})
            self._WrapDataVar('Q', 'ADC units', 'normal', {'linestyle': 'g-'})
            self._WrapDataVar('I Std Dev', 'ADC units', 'std')
            self._WrapDataVar('Q Std Dev', 'ADC units', 'std')
            self._WrapDataVar('ADC Amplitude', 'ADC units', None, {'linestyle': 'r-'})
            self._WrapDataVar('ADC Phase', 'rad', None, {'linestyle': 'k-'})
            self._WrapExptVar('Rep Iteration')

            run_data['Single Shot Is'] = run_data['I']
            run_data['Single Shot Qs'] = run_data['Q']
            run_data['I'] = np.mean(run_data['Single Shot Is'])
            run_data['Q'] = np.mean(run_data['Single Shot Qs'])
            run_data['I Std Dev'] = np.std(run_data['Single Shot Is'])
            run_data['Q Std Dev'] = np.std(run_data['Single Shot Qs'])
            run_data['ADC Amplitude'] = np.sqrt(run_data['I']**2 + run_data['Q']**2)
            run_data['ADC Phase'] = np.arctan2(run_data['Q'], run_data['I']) # numpy.arctan2(y, x) expects reversed arguments.
            
            extra_data['Indep Names'] = [['Rep Iteration']]
            extra_data['Indep Vals'] = [[np.linspace(1, len(Is), len(Is))]]
            extra_data['Dependencies'] = {'Single Shot Is': extra_data['Indep Names'][0],
                                          'Single Shot Qs': extra_data['Indep Names'][0]}
        elif self.ADCSettings[self.ADCs.index(ADCName)]['RunMode'] == 'average':
            self._WrapDataVar('Software Demod I', 'ADC units', None, {'linestyle': 'b-'})
            self._WrapDataVar('Software Demod Q', 'ADC units', None, {'linestyle': 'g-'})
            self._WrapDataVar('Software Demod ADC Amplitude', 'ADC units', None, {'linestyle': 'r-'})
            self._WrapDataVar('Software Demod ADC Phase', 'rad', None, {'linestyle': 'k-'})
            self._WrapExptVar('Reps', '', 1)
            self._WrapExptVar('ADC Time', 'ns')
            
            time = np.linspace(0, 2 * (len(Is) - 1), len(Is))
            run_data['Software Demod I'], run_data['Software Demod Q'] = self._SoftwareDemodulate(time, run_data['I'], run_data['Q'], ADCName)
            run_data['Software Demod ADC Amplitude'] = np.sqrt(run_data['Software Demod I']**2 + run_data['Software Demod Q']**2)
            run_data['Software Demod ADC Phase'] = np.arctan2(run_data['Software Demod Q'], run_data['Software Demod I']) # numpy.arctan2(y, x) expects reversed arguments.

            extra_data['Indep Names'] = [['ADC Time']]
            extra_data['Indep Vals'] = [[time]]
            extra_data['Dependencies'] = {'I': extra_data['Indep Names'][0], 'Q': extra_data['Indep Names'][0]}
        
        if self.ADCSettings[self.ADCs.index(ADCName)]['RunMode'] in ['average', 'demodulate']:
            return run_data, extra_data
        else:
            return run_data, None

    def _AverageData(self, data, extra_data):
        """
        This method should be used for proper averaging of the data returned by RunOnce method.
        
        The method should return a "data" and an "extra data" dictionaries.
        
        Inputs: 
            Runs: number of independent runs of the experiment.
        """
        avg_data = {}
        for key in data:
            if key in ['I', 'Q']:
                if 'Single Shot ' + key + 's' in data:
                    avg_data[key] = np.mean(data['Single Shot ' + key + 's'])
                    avg_data[key + ' Std Dev'] = np.std(data['Single Shot ' + key + 's'])
                else:
                    self._WrapDataVar(key, self._GetUnits(key), 'normal')
                    self._WrapDataVar(key + ' Std Dev', self._GetUnits(key), 'std')
                    avg_data[key] = np.mean(data[key], axis=0)            
                    avg_data[key + ' Std Dev'] = np.std(data[key], axis=0)
            if key in ['Software Demod I', 'Software Demod Q']:
                self._WrapDataVar(key, self._GetUnits(key), 'normal')
                self._WrapDataVar(key + ' Std Dev', self._GetUnits(key), 'std')
                avg_data[key] = np.mean(data[key], axis=0)            
                avg_data[key + ' Std Dev'] = np.std(data[key], axis=0)
        
        for key in data:      
            if key == 'ADC Amplitude':
                avg_data['ADC Amplitude'] = np.sqrt(avg_data['I']**2 + avg_data['Q']**2)
            if key == 'ADC Phase':
                avg_data['ADC Phase'] = np.arctan2(avg_data['Q'], avg_data['I']) # numpy.arctan2(y, x) expects reversed arguments.
            
            if key == 'Software Demod ADC Amplitude':
                avg_data['Software Demod ADC Amplitude'] = np.sqrt(avg_data['Software Demod I']**2 + avg_data['Software Demod Q']**2)
            if key == 'Software Demod ADC Phase':
                avg_data['Software Demod ADC Phase'] = np.arctan2(avg_data['Software Demod Q'], avg_data['Software Demod I']) # numpy.arctan2(y, x) expects reversed arguments.

        return avg_data, extra_data