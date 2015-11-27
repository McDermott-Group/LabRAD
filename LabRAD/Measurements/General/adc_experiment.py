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

import sys
import numpy as np
import matplotlib.pyplot as plt

import labrad.units as units

import experiment as expt
import data_processing as dp


class ADCExperiment(expt.Experiment):        
    def single_shot_iqs(self, adc=None, save=False, plot_data=None):
        """
        Run a single experiment, saving individual I and Q points.
        
        Inputs:
            adc: ADC board name. If the board is not specified
                and there is only one board in experiment resource 
                dictionary than it will be used by default.
            save: if True save the data (default: False).
            plot_data: if True plot the data (default: True).
        Output:
            None.
        """
        adc = self.boards.get_adc(adc)
        previous_adc_mode = self.boards.get_adc_setting('RunMode', adc)
        self.boards.set_adc_setting('RunMode', 'demodulate', adc)
        
        self.load_once()
        data = self._process_data(self.run_once())
        
        self.boards.set_adc_setting('RunMode', previous_adc_mode, adc)
        
        if plot_data is not None:
            self._plot_iqs(data)

        # Save the data.
        if save:
            self._save_data(data)

    def single_shot_osc(self, adc=None, save=False, plot_data=None):
        """
        Run a single shot experiment in average mode, and save the 
        time-demodulated data to file.
        
        Inputs: 
            adc: ADC board name. If the board is not specified
                and there is only one board in experiment resource
                dictionary than it will be used by default.
            save: if True save the data if save is True (default: False).
            plot_data: data variables to plot (default: None).
        Output:
            None.
        """
        self.avg_osc(adc, save, plot_data, runs=1)

    def avg_osc(self, adc=None, save=False, plot_data=None, runs=100):
        """
        Run a single experiment in average mode specified number of
        times and average the results together.
        
        Inputs:
            adc: ADC board name.
            save: if True save the data if save is True (default: False).
            plot_data: data variables to plot.
            runs: number of runs (default: 100).
        Output:
            None.
        """
        self._sweep_status = ''
        self._sweep_msg = ''
        
        adc = self.boards.get_adc(adc)
        previous_adc_mode = self.boards.get_adc_setting('RunMode', adc)
        self.boards.set_adc_setting('RunMode', 'average', adc)
        
        if self.value('Reps') is not None:
            prev_reps = self.value('Reps')
        
        self.load_once()
        data = self._process_data(self.run_once())

        # Make a list of data variables that should be plotted.
        if plot_data is not None:
            for var in self._comb_strs(plot_data):
                if var not in data:
                    print("Warning: variable '" + var + 
                    "' is not found among the data dictionary keys: " + 
                    str(data.keys()) + ".")
            plot_data = [var for var in
                    self._comb_strs(plot_data) if var in data]
        if plot_data:
            self._init_1d_plot([['Time']], [[data['Time']['Value']]],
                    data, plot_data)
 
        if runs > 1:        # Run multiple measurements (shots).
            print('\n\t[ESC]:\tAbort the run.' + 
                  '\n\t[S]:\tAbort the run but [s]ave the data.\n')
            self.add_var('Runs', runs)
            sys.stdout.write('Progress: %5.1f%%\r' %(100. / runs))
            data_to_plot = {}
            for key in data:
                data_to_plot[key] = data[key].copy()
            for r in range(runs - 1):
                # Check if the specific keys are pressed.
                self._listen_to_keyboard(recog_keys=[27, 83, 115], 
                        clear_buffer=False)
                if self._sweep_status in ['abort', 'abort-and-save']:
                    self.value('Runs', r + 1)
                    print(self._sweep_msg)
                    break  
                run_data = self.run_once()
                sys.stdout.write('Progress: %5.1f%%\r' %(100. * (r + 2) / runs))
                for key in data:
                    if data[key]['Type'] == 'Dependent':
                        # Accumulate the data values.
                        # These values should be divided by the actual
                        # number of Reps to get the average values.
                        data[key]['Value'] = (data[key]['Value'] +
                                          run_data[key]['Value'])
                if plot_data and np.mod(r, 10) == 0:
                    for key in plot_data:
                        data_to_plot[key]['Value'] = (data[key]['Value'] /
                                                      float(r + 1))
                    self._update_1d_plot([['Time']],
                            [[data['Time']['Value']]], data_to_plot,
                            plot_data, np.size(data['Time']['Value']) - 1)
            for key in data:
                if 'Value' in data[key] and data[key]['Type'] == 'Dependent':
                    data[key]['Value'] = data[key]['Value'] / float(self.value('Runs'))
        
        if plot_data:        # Refresh the plot.
            self._update_1d_plot([['Time']], [[data['Time']['Value']]],
                    data, plot_data, np.size(data['Time']['Value']) - 1)
        
        self.boards.set_adc_setting('RunMode', previous_adc_mode, adc)

        # Save the data.
        if ((save and self._sweep_status != 'abort') or
                self._sweep_status == 'abort-and-save'):
            self._save_data(data)
        
        # Restore the original state of some special variables.
        self.rm_var('Runs')
        if 'prev_reps' in locals(): 
            self.add_var('Reps', prev_reps)
        
        print('The data collection has been succesfully finished.')
        
    def _plot_iqs(self, data):
        plt.ion()
        plt.figure(13)
        plt.plot(data['Is']['Value'], data['Qs']['Value'], 'b.')
        plt.xlabel('I [ADC Units]')
        plt.ylabel('Q [ADC Units]')
        plt.title('Single Shot Is and Qs')
        plt.axis('equal')
        plt.draw()
        plt.pause(0.05)

    def init_expt(self):
        # This solves a potential issue when 'Reps' are overwritten
        # by the run_once or load_once methods.
        if self.value('Reps') is not None:
            self._prev_reps = self.value('Reps')
    
    def exit_expt(self):
        if hasattr(self, '_prev_reps'):
            self.value('Reps', self._prev_reps)
        
    def run_once(self, adc=None):
        self.get('Temperature')
        result = self.boards.run(self.value('Reps'))
        Is, Qs = result[0] 
        Is = np.array(Is)
        Qs = np.array(Qs)
                
        if self.boards.get_adc_setting('RunMode', adc) == 'demodulate':
            I = np.mean(Is)
            Q = np.mean(Qs)
            As = np.hypot(Is, Qs)
            return {
                    'Is': { 
                        'Value': Is * units.ADCUnits,
                        'Dependencies': 'Repetition Index',
                        'Preferences':  {'linestyle': 'b.'}},
                    'Qs': { 
                        'Value': Qs * units.ADCUnits,
                        'Dependencies': 'Repetition Index',
                        'Preferences':  {'linestyle': 'g.'}},
                    'Amplitudes': { 
                        'Value': As * units.ADCUnits,
                        'Dependencies': 'Repetition Index',
                        'Preferences':  {'linestyle': 'r.'}},
                    'Phases': { # numpy.arctan2(y, x) expects reversed arguments.
                        'Value': np.arctan2(Qs, Is) * units.rad,
                        'Dependencies': 'Repetition Index',
                        'Preferences':  {'linestyle': 'k.'}},
                    'I': {
                        'Value': I * units.ADCUnits,
                        'Distribution': 'normal',
                        'Preferences':  {'linestyle': 'b-'}},
                    'Q': { 
                        'Value': Q * units.ADCUnits,
                        'Distribution': 'normal',
                        'Preferences':  {'linestyle': 'g-'}}, 
                    'I Std Dev': { 
                        'Value': np.std(Is) * units.ADCUnits},
                    'Q Std Dev': { 
                        'Value': np.std(Qs) * units.ADCUnits},
                    'Amplitude': { 
                        'Value': np.hypot(I, Q) * units.ADCUnits,
                        'Preferences':  {'linestyle': 'r-'}},
                    'Phase': { # numpy.arctan2(y, x) expects reversed arguments.
                        'Value': np.arctan2(Q, I) * units.rad,
                        'Preferences':  {'linestyle': 'k-'}},
                    'Mean Absolute Amplitude': { 
                        'Value': np.mean(As) * units.ADCUnits,
                        'Preferences':  {'linestyle': 'm-'}},
                    'Mean Absolute Amplitude Std Dev': { 
                        'Value': np.std(As) * units.ADCUnits},
                    'Repetition Index': {
                        'Value': np.linspace(1, len(Is), len(Is)),
                        'Type': 'Independent'},
                    'Temperature': {
                        'Value': self.acknowledge_request('Temperature')}
                   }
        elif self.boards.get_adc_setting('RunMode', adc) == 'average':
            self.value('Reps', 1)
            time = np.linspace(0, 2 * (len(Is) - 1), len(Is))
            I, Q = dp.software_demod(time, self.boards.get_adc_setting('DemodFreq', adc), Is, Qs)
            return {
                    'I': { 
                        'Value': Is * units.ADCUnits,
                        'Dependencies': 'Time',
                        'Preferences':  {'linestyle': 'b-'}},
                    'Q': { 
                        'Value': Qs * units.ADCUnits,
                        'Dependencies': 'Time',
                        'Preferences':  {'linestyle': 'g-'}},
                    'Software Demod I': { 
                        'Value': I * units.ADCUnits,
                        'Preferences':  {'linestyle': 'b.'}},
                    'Software Demod Q': { 
                        'Value': Q * units.ADCUnits,
                        'Preferences':  {'linestyle': 'g.'}}, 
                    'Software Demod Amplitude': { 
                        'Value': np.hypot(I, Q) * units.ADCUnits,
                        'Preferences':  {'linestyle': 'r.'}},
                    'Software Demod Phase': { 
                        'Value': np.arctan2(Q, I) * units.rad,
                        'Preferences':  {'linestyle': 'k.'}},
                    'Time': {
                        'Value': time * units.ns,
                        'Type': 'Independent'},
                    'Temperature': {
                        'Value': self.acknowledge_request('Temperature')}
                   }

    def average_data(self):
        """
        Average the data acquired by method run_n_times.
        """
        data = self._run_n_data
        if self._sweep_pts_acquired == 0:
            self._avg_data = {key: data[key].copy() for key in data}

        avg = self._avg_data
        if 'Is' in data and 'Qs' in data:
            Is = self.strip_units(self._avg_data['Is']['Value'])
            Qs = self.strip_units(self._avg_data['Qs']['Value'])
            avg['I']['Value'] = np.mean(Is) * units.ADCUnits
            avg['Q']['Value'] = np.mean(Qs) * units.ADCUnits
            avg['I Std Dev']['Value'] = np.std(Is) * units.ADCUnits
            avg['Q Std Dev']['Value'] = np.std(Qs) * units.ADCUnits
            
            As = np.hypot(Is, Qs)
            avg['Mean Absolute Amplitude']['Value'] = np.mean(As) * units.ADCUnits
            avg['Mean Absolute Amplitude Std Dev']['Value'] = np.std(As) * units.ADCUnits
            
            I = self.strip_units(avg['I']['Value'])
            Q = self.strip_units(avg['Q']['Value'])
            avg['Amplitude']['Value'] = np.hypot(I, Q) * units.ADCUnits
            avg['Phase']['Value'] = np.arctan2(Q, I) * units.rad
        else:
            It = self.strip_units(data['I']['Value'])
            Qt = self.strip_units(data['Q']['Value'])
            avg['I']['Value'] = np.mean(It, axis=0) * units.ADCUnits
            avg['Q']['Value'] = np.mean(Qt, axis=0) * units.ADCUnits
            avg['I']['Distribution'] = 'normal'
            avg['Q']['Distribution'] = 'normal'
            avg['I Std Dev']['Value'] = np.std(It, axis=0) * units.ADCUnits
            avg['Q Std Dev']['Value'] = np.std(Qt, axis=0) * units.ADCUnits

            Id = self.strip_units(data['Software Demod I']['Value'])
            Qd = self.strip_units(data['Software Demod Q']['Value'])
            avg['Software Demod I']['Value'] = np.mean(Id) * units.ADCUnits
            avg['Software Demod Q']['Value'] = np.mean(Qd) * units.ADCUnits
            avg['Software Demod I']['Distribution'] = 'normal'
            avg['Software Demod Q']['Distribution'] = 'normal'
            avg['Software Demod I Std Dev'] = {'Value': np.std(Id) * units.ADCUnits}
            avg['Software Demod Q Std Dev'] = {'Value': np.std(Qd) * units.ADCUnits}
            
            avg['Software Demod Amplitude']['Value'] = np.hypot(Id, Qd) * units.ADCUnits
            avg['Software Demod Phase']['Value'] = np.arctan2(Qd, Id) * units.rad

        if 'Temperature' in data:
            T = self.strip_units(data['Temperature']['Value'])
            avg['Temperature']['Value'] = (np.mean(T) *
                    self.unit_factor(data['Temperature']['Value']))