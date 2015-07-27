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

import data_processing

DAC_ZERO_PAD_LEN = 20

class HEMTQubitReadout(expt.Experiment):
    """
    Read out a qubit connected to a resonator.
    """
    def RunOnce(self, ADCName=None, plot_waveforms=False):
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
        RO_SB_freq = self.value('Readout SB Frequency')['GHz']       # readout sideband frequency (RO_SB_freq in GHz)
        RO_amp = self.value('Readout Amplitude')['DACUnits']         # amplitude of the sideband modulation
        RO_time = self.value('Readout Time')['ns']                   # length of the readout pulse
        
        #QUBIT DRIVE VARIABLES#####################################################################
        QB_SB_freq = self.value('Qubit SB Frequency')['GHz']         # qubit sideband frequency (RO_SB_freq in GHz)
        QB_amp = self.value('Qubit Amplitude')['DAC units']          # amplitude of the sideband modulation
        QB_time = self.value('Qubit Time')['ns']                     # length of the qubit pulse
      
        #TIMING VARIABLES##########################################################################
        QBtoRO = self.value('Qubit Drive to Readout Delay')['ns']    # delay from the start of the qubit pulse to the start of the readout pulse
        ADC_wait_time = self.value('ADC Wait Time')['ns']            # delay from the start of the readout pulse to the start of the demodulation
        
        ###WAVEFORMS###############################################################################
        requested_waveforms = [settings[ch] for settings in
                self.fpga_boards.dac_settings for ch in ['DAC A', 'DAC B']]

        waveforms = {};
        if 'None' in requested_waveforms:
            waveforms['None'] = np.hstack([pulse.DC(2 * DAC_ZERO_PAD_LEN + QB_time + QBtoRO + RO_time, 0)])
        
        if 'Readout I' in requested_waveforms:
            waveforms['Readout I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.CosinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])

        if 'Readout Q' in requested_waveforms:
            waveforms['Readout Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN + QB_time + QBtoRO, 0),
                                                pulse.SinePulse(RO_time, RO_SB_freq, RO_amp, 0.0, 0.0),
                                                pulse.DC(DAC_ZERO_PAD_LEN, 0)])
 
        if 'Qubit I' in requested_waveforms:            
            waveforms['Qubit I'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.CosinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])

        if 'Qubit Q' in requested_waveforms:        
            waveforms['Qubit Q'] = np.hstack([pulse.DC(DAC_ZERO_PAD_LEN, 0),
                                              pulse.SinePulse(QB_time, QB_SB_freq, QB_amp, 0.0, 0.0),
                                              pulse.DC(QBtoRO + RO_time + DAC_ZERO_PAD_LEN, 0)])
 
        for idx, settings in enumerate(self.fpga_boards.dac_settings):
            for channel in ['DAC A', 'DAC B']:
                if self.fpga_boards.dac_settings[idx][channel] not in waveforms:
                    raise expt.ResourceDefinitionError("'" + 
                        str(self.fpga_boards.dacs[idx]) + 
                        "' setting '" + str(channel) + "': '" +
                        self.fpga_boards.dac_settings[idx][channel] +
                        "' could not be recognized. The allowed '" +
                        str(channel) + "' values are 'Readout I', '" +
                        "Readout Q', 'Qubit I', 'Qubit Q', and 'None'.")

        if plot_waveforms:
            self._plot_waveforms([waveforms[wf] for wf in requested_waveforms],
                    ['r', 'g', 'b', 'k'], requested_waveforms)

        SRAMLength = len(waveforms[self.DACSettings[0]['DAC A']])
        SRAMDelay = np.ceil(SRAMLength / 1000)
                              
        ADCName = self._GetADCName(ADCName)
        self.ADCSettings[self.ADCs.index(ADCName)]['ADCDelay'] = (DAC_ZERO_PAD_LEN +
                ADC_wait_time + QB_time + QBtoRO) * units.ns        # Waiting time before starting demodulation.
        self.ADCSettings[self.ADCs.index(ADCName)]['DemodFreq'] = -self.value('Readout SB Frequency')

        DAC1_SRAM = dac.waves2sram(waveforms[self.DACSettings[0]['DAC A']], waveforms[self.DACSettings[0]['DAC B']])
        DAC2_SRAM = dac.waves2sram(waveforms[self.DACSettings[1]['DAC A']], waveforms[self.DACSettings[1]['DAC B']])
        DAC_mem =  dac.mem_simple(self.value('Init Time')['us'], SRAMLength, 0, SRAMDelay)
        
        ###RUN#####################################################################################
        self.acknowledge_requests()
        P = self.fpga_boards.load_and_run(waveforms, [mem_list1, mem_list2], self.value('Reps'))
        
        ###DATA POST-PROCESSING####################################################################
        Is, Qs = result[0] 
        
        run_data = {'I': np.array(Is),
                    'Q': np.array(Qs)}

        if self.ADCSettings[self.ADCs.index(ADCName)]['RunMode'] == 'demodulate':
            self._WrapDataVar('Single Shot Is', 'ADC units', None, {'linestyle': 'b.'})
            self._WrapDataVar('Single Shot Qs', 'ADC units', None, {'linestyle': 'g.'})
            self._WrapDataVar('I', 'ADC units', 'normal', {'linestyle': 'b-'})
            self._WrapDataVar('Q', 'ADC units', 'normal', {'linestyle': 'g-'})
            self._WrapDataVar('I Std Dev', 'ADC units', 'std')
            self._WrapDataVar('Q Std Dev', 'ADC units', 'std')
            self._WrapDataVar('ADC Amplitude', 'ADC units', None, {'linestyle': 'r-'})
            self._WrapDataVar('ADC Phase', 'rad', None, {'linestyle': 'k-'})
            self.value('Rep Iteration')

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
            self.value('Reps', '', 1)
            self.value('ADC Time', 'ns')
            
            time = np.linspace(0, 2 * (len(Is) - 1), len(Is))
            run_data['Software Demod I'], run_data['Software Demod Q'] = self._SoftwareDemodulate(time, run_data['I'], run_data['Q'], ADCName)
            run_data['Software Demod ADC Amplitude'] = np.sqrt(run_data['Software Demod I']**2 + run_data['Software Demod Q']**2)
            run_data['Software Demod ADC Phase'] = np.arctan2(run_data['Software Demod Q'], run_data['Software Demod I']) # numpy.arctan2(y, x) expects reversed arguments.

            extra_data['Indep Names'] = [['ADC Time']]
            extra_data['Indep Vals'] = [[time]]
            extra_data['Dependencies'] = {'I': extra_data['Indep Names'][0], 'Q': extra_data['Indep Names'][0]}
        
        ###DATA VARIABLES####################################################################################
        #####################################################################################################
        # Units for data variables as well as plotting preferences can be defined here.
        # Example: self._WrapDataVar('P',  '', 'binomial', ' {'name': 'Probability', 'linestyle': 'b-', 'linewidth': 2, 'legendlabel': 'Prob.', 'ylim': [0, 1]})
        self._WrapDataVar('I', 'ADC units', None, {'linestyle': 'b-', 'linewidth': 1})
        self._WrapDataVar('Q', 'ADC units', None, {'linestyle': 'g-', 'linewidth': 1})

        
        if self.ADCSettings[self.ADCs.index(ADCName)]['RunMode'] in ['average', 'demodulate']:
            return run_data, extra_data
        else:
            return run_data

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