# Copyright (C) 2015 Guilhem Ribeill, Ivan Pechenezhskiy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Base class for all python-based LabRAD experiments. This class should be
the parent class for each particular experiment. It provides shared 
functionality for setting experiment variables, electronics 
initialization, running 1D- and 2D-sweeps, and saving data to .txt 
and .mat files.

A basic experiment program would look something like this:

import os
import numpy as np

from labrad.units import GHz, dBm

import some_experiment

information = { 
                'Device Name': 'Test Device',
                'User': 'Test User',
                'Base Path': 'Z:\mcdermott-group\Data\Test',
                'Experiment Name': 'Test_Experiment',
                'Comments': 'This is only a test.' 
              }
resources =  { # GPIB RF Generator.
                'Interface': 'RF Generator',
                'Server': 'GPIB RF Generators',
                'Address': (os.environ['COMPUTERNAME'].lower() + 
                            ' GPIB Bus - GPIB0::19::INSTR)',
                'Variables': { 
                                'Power': {'Setting': 'Power'}, 
                                'Frequency': {'Setting': 'Frequency'}
                             }
              },

variables = {
                'RF Frequency' : 10 * GHz,
                'RF Power': 13 * dBm
            }

with some_experiment.SomeExperiment() as expt:    
    expt.set_experiment(information, resources, variables) 
    freq = np.linspace(2, 5, 101) * GHz
    expt.sweep('RF Frequency', freq, save=True)
"""

import os
import sys
import time
import warnings
from msvcrt import kbhit, getch

import numpy as np
import matplotlib
try:
    matplotlib.use('GTKApp')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import scipy.io as sio

import labrad
import labrad.units as units

import server_interfaces


class ExperimentDefinitionError(Exception):
    """Experiment specification error."""
    pass


class DataError(Exception):
    """Data specification error."""
    pass
    

class SweepError(Exception):
    """Data sweep errro."""
    pass


class Experiment(object):
    """
    Experiment class. Parent class for specific instances of experiments
    that provides shared functionality.
    """
###SPECIAL METHODS######################################################
    def __init__(self):
        """
        Input:
            None.
        Output:
            None.
        """
        print('Connecting to the LabRAD manager...')
        password = None
        # Open the LabRAD initialization file.
        script_path = os.path.dirname(__file__)
        labrad_ini_file = os.path.join(script_path.rsplit('LabRAD', 1)[0],
                'LabRAD', 'LabRAD.ini')
        if os.path.isfile(labrad_ini_file):
            with open(labrad_ini_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                if line.find('Password: ') != -1:
                    password = line.split('Password: ')[-1]
                    break
        try:
            self.cxn = labrad.connect(password=password)
        except:
            raise Exception('Connection to the LabRAD manager could '+
                    'not be established.')
        # This flag controls the standard output upon pressing [O]
        # during an experiment sweep.
        self._standard_output = True

    def __del__(self):
        """Just in case..."""
        plt.close('all')

    def __enter__(self):
        """
        Context entry. For now all it does is return a copy of 
        the initialized class.
        """
        return self
    
    def __exit__(self, type, value, traceback):
        """
        Safely disconnect from the LabRAD manager.
        Catch exceptions if needed.
        """
        # Close the resources/interfaces properly.
        if hasattr(self, '_vars'):
            for var in self._vars:
                if ('Interface' in self._vars[var] and
                        hasattr(self._vars[var]['Interface'], '__exit__')):
                    self._vars[var]['Interface'].__exit__(type, value, traceback)
        
        # Disconnect from LabRAD.
        if hasattr(self, 'cxn'):
            self.cxn.disconnect()
        
        # Delete empty folders.
        if hasattr(self, '_save_path'):
            for subdir in ['TextData', 'MATLABData']:
                subpath = os.path.join(self._save_path, subdir)
                if os.path.exists(subpath) and not os.listdir(subpath):
                    os.rmdir(subpath)
            for idx in range(5):
                subpath = os.path.split(subpath)[0]
                if os.path.exists(subpath) and not os.listdir(subpath):
                    try:
                        os.rmdir(subpath)
                    except WindowsError:
                        pass

        print('The instrument resources have been safely terminated! ' + 
              'Have a nice day.')
  
    ###SETUP METHODS####################################################
    def set_experiment(self, information, resources, variables):
        """
        Set experiment information, resources and experiment variables.
        
        Inputs:
            information: dictionary of the following format:
                {'Device Name': 'name of the resource under study',
                 'User': 'who is running the experiment?',
                 'Base Path': 'initial path for data saving purposes',
                 'Experiment Name': 'what are you doing?'
                 'Comments': 'any comments that will go in the header 
                        of each saved data file'}.
                 Key 'Comments' is optional and will be set to '',
                 if not included. The data will be saved in
                 "Base Path\User\Device\Date\Experiment Name\".
            resources: list of resources in the following format:
              [ { # Generic resource.
                    'Interface': 'ResourceServerInterface',
                    'Device Address': 'address that could be used to
                        select a specific device',
                    'Variables': {
                                    'Variable 1': {'Setting': 
                                    'server_setting_that_controls_Variable_1',
                                    'Value': some_default_value(optional)}, 
                                    'Variable 2': {'Setting':
                                    'server_setting_that_controls_Variable_2'}
                                 }
                },
                { # Waveform parameters.
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'Shasta Board DAC 9', 
                            'Shasta Board DAC 10',
                            'Shasta Board ADC 11'
                          ],
                'Shasta Board DAC 9':  {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'Data': True
                                       },
                'Shasta Board DAC 10': {   
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I',
                                       },
                'Shasta Board ADC 11':  {
                                        'RunMode': 'demodulate',
                                        'FilterType': 'square',
                                        'FilterStartAt': 0 * ns,
                                        'FilterWidth': 9500 * ns,
                                        'FilterLength': 10000 * ns,
                                        'FilterStretchAt': 0 * ns,
                                        'FilterStretchLen': 0 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -30 * MHz,
                                        'ADCDelay': 0 * ns,
                                        'Data': False
                                       },
                'Variables': {
                                'Init Time': {},
                                'Qubit SB Frequency': {'Value': 0 * MHz},
                                'Qubit Amplitude': {'Value': 0 * DACUnits},
                                'Qubit Time': {'Value': 0 * ns},
                                'ADC Wait Time': {'Value': 0 * ns}
                             }
                },
                { # GPIB RF Generator.
                    'Interface': 'GPIB RF Generator'
                    'Address': 'computer-name GPIB Bus - GPIB0::10:INSTR',
                    'Variables': {
                                    'Qubit Power': {'Setting': 'Power'}, 
                                    'Qubit Frequency': {'Setting': 'Frequency'}
                                 }
                },
                { # Lab Brick Attenuator.
                    'Interface': 'Lab Brick Attenuator',
                    'Serial Number': 7032,
                    'Variables': 'RF Attenuation'
                { # SIM Voltage Source.
                    'Interface': 'SIM928 Voltage Source',
                    'Address': 'GPIB0::10::SIM900::3',
                    'Variables': 'Bias Voltage'
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Reps',
                                  'Actual Reps',
                                  'Threshold'],
                }
              ]
            variables: experiment variable dictionary 
                in the {'Variable Name': Value,...} format.
        Output:
            None.
        """
        # Check that all variables are in information as expected.   
        if 'Device Name' not in information:
            raise ExperimentDefinitionError('Device name is not specified.')
        if 'User' not in information:
            raise ExperimentDefinitionError('User is not specified.')
        if 'Base Path' not in information:
            raise ExperimentDefinitionError('Base path is not specified.')
        if 'Experiment Name' not in information:
            raise ExperimentDefinitionError('Experiment name is not specified.')
        if 'Comments' not in information:
            information['Comments'] = ''

        self.information = information
        
        # Get today's date in MM-DD-YY format, and make the save path.
        today = time.strftime("%m-%d-%y", time.localtime())
        self._save_path = os.path.join(information['Base Path'],
                                       information['User'],
                                       information['Device Name'],
                                       today,
                                       information['Experiment Name'])
        
        # Make the save path if it does not exist.
        if not os.path.exists(self._save_path):
            try:
                os.makedirs(self._save_path)
            except:
                raise IOError('Could not create experiment save path!' + 
                              ' Is AFS on?')

        # Set electronics information (which DACs, ADCs, and RF
        # generators, Lab Bricks, etc.), initialize connections
        # to the LabRAD servers, create a map of the experiment
        # variables (defined in resources) to the resources.
        self._vars = {}     # Dictionary of the experiment variables.
        
        def _msg(key, resource):
            return ("'" + key + "' key is not found in the resource " + 
                   " dictionary: " + str(resource) + ".")
        
        for res in resources:
            if 'Interface' not in res:
                raise ExperimentDefinitionError(_msg('Interface', res))
            
            if 'Variables' not in res:
                raise ExperimentDefinitionError(_msg('Variables', res))
            elif (not isinstance(res['Variables'], str) and 
                    not isinstance(res['Variables'], list) and
                    not isinstance(res['Variables'], dict)):
                raise ExperimentDefinitionError("'Variables' key" +
                " in the resource dictionary: " + str(res) +
                " should be defined as a list of experiment" + 
                " variables or as a simple string for a single variable.")
            if isinstance(res['Variables'], str):
                res['Variables'] = [res['Variables']]
            
            var_res = res.copy()
            var_res.pop('Variables', None)
            if isinstance(res['Variables'], list):
                for var in res['Variables']:
                    self._vars[var] = var_res.copy()
            elif isinstance(res['Variables'], dict):
                for var, var_dict in res['Variables'].items():
                    self._vars[var] = var_res.copy()
                    if isinstance(var_dict, dict):
                        for property in var_dict:
                            self._vars[var][property] = res['Variables'][var][property]
                    else:
                        raise ExperimentDefinitionError("Variable " +
                                "properties in the resource dictionary " +
                                str(res) + " should be specified " + 
                                "in a dictionary.")
            else:
                raise ExperimentDefinitionError("Variables in the" +
                    " resource dictionary " + str(res) +
                    " should be specified as a list of strings," + 
                    " a dictionary," + 
                    " or as a string for a single variable.")

            # Readings entered manually, software parameters.
            if res['Interface'] is None:
                pass
            else:
                interface = res['Interface'].replace(' ', '')
                # GHz FPGA boards.
                if interface == 'GHzFPGABoards':
                    self.boards = getattr(server_interfaces, interface)(self.cxn, res)
                # Resources specified in module sever_interfaces: Lab Brick
                # attenuators, RF generators, voltage sources, etc.
                elif hasattr(server_interfaces, interface):
                    for var in res['Variables']:
                        self.interface(var, res)
                else:
                    print("Warning: resource type '" + str(res['Interface']) +
                          "' is not yet supported.")

        # Set experiment variables.
        for var in variables:
            if var in self._vars:
                self.value(var, variables[var])
            else:
                print("Warning: variable '" + str(var) + 
                      "' is not found in the experiment variables. " +
                      "Unless the variable is explicitly used, " + 
                      "its value will not be saved.")
                self._vars[var] = {'Value': variables[var],
                                   'Save': False}
              
    def _check_var(self, var, check_type=True,
                              check_exist=True,
                              check_value=True,
                              check_interface=True):
        """
        Assert the existence of an experiment variable.
        
        Inputs:
            var: name of the experiment variable.
            check_type (optional): check whether the var type is string
                (default: True).
            check_exist (optional): check whether the var exist among
                the experiment variables (default: True).
            check_value (optional): check whether the var value is
                defined (default: True).
            check_interface (optional): check whether the var interface
                is defined (default: True).
        Output:
            None.
        """        
        if check_type and not isinstance(var, str):
            raise ExperimentDefinitionError("'" + str(var) + "' is " +
                    "expected to be a string that defines an " + 
                    "experiment variable.")
        if check_exist and var not in self._vars:
            raise ExperimentDefinitionError("'" + str(var) + 
                    "' variable is not found among the experiment " +
                    "variables: " + str(self._vars.keys()) + ".")
        if check_value and 'Value' not in self._vars[var]:
            raise ExperimentDefinitionError("No value is assigned to " + 
                    "the experiment variable '" + str(var) + "'.")
        if check_interface and 'Interface' not in self._vars[var]:
            raise ExperimentDefinitionError("No resource " + 
                    "is responsible for the experiment variable '" +
                    str(var) + "'.")
    
    def add_var(self, var, value=None):
        """
        Register a new experiment variable.
        
        Inputs:
            var: name of the experiment variable.
            value (optional): a new value to assign to the experiment 
                variable.
        Output:
            value: value of the experiment variable.
        """
        if not hasattr(self, '_vars'):
            raise ExperimentDefinitionError('Experiment resources ' + 
            'should be set prior accessing any variables.')
        
        if var in self._vars:
            if value is not None:
                self._vars[var]['Value'] = value
                self._vars[var]['Save'] = True
        elif value is not None:
            self._vars[var] = {'Value': value, 'Save': True}
        else:
            self._vars[var] = {'Value': None, 'Save': False}

        return self._vars[var]['Value']
        
    def rm_var(self, var):
        """
        Remove an experiment variable.
        
        Input:
            var: name of the experiment variable.
        Output:
            None
        """
        if hasattr(self, '_vars'):
            self._vars.pop(var, None)
        
    def interface(self, var, res=None):
        """
        Get or set the interface responsible for a given variable.
        
        Input:
            var: name of the experiment variable.
            res: resource dictionary.
        Output:
            interface: interface responsible for the variable.
        """
        if res is not None:
            self._check_var(var, check_value=False, check_interface=False)
            self._vars[var]['Interface'] = getattr(server_interfaces,
            res['Interface'].replace(' ', ''))(self.cxn, res, var)
        else:
            self._check_var(var, check_value=False)
        return self._vars[var]['Interface']
        
    def set(self, var, value=None):
        """
        Send a non-blocking request to a server to set an experiment
        variable.
        
        Inputs: 
            var: variable name.
            value (optional): new variable value.
        Output: 
            None.
        """
        try:
            interface = self.interface(var)
        except:
            return None
        if interface is not None and hasattr(interface, 'send_request'):
            if value is None:
                interface.send_request(self.value(var))
            else:
                # Save the current variable value, just in case if 
                # a correction to the value was made.
                # This is useful when a side-band frequency offset or
                # a bias voltage offset is added but the orginal number
                # is a more preferable value to refer to.
                prev_val = self.value(var)
                interface.send_request(self.value(var, value))
                # Restore the previous value.
                self.value(var, prev_val)
            
    def get(self, var):
        """
        Send a non-blocking get request to a server.
        
        Input: 
            var: variable or setting name
        Output: 
            None.
        """
        try:
            interface = self.interface(var)
        except:
            return None
        if interface is not None and hasattr(interface, 'send_request'):
            interface.send_request(None)
        
    def acknowledge_request(self, var):
        """
        Wait for the result of a non-blocking request to set a variable.
        
        Input: 
            var: variable name.
        Output: 
            result: result of a request obtained from a server.
        """
        try:
            interface = self.interface(var)
        except:
            return None
        if interface is not None and hasattr(interface, 'acknowledge_request'):
            return interface.acknowledge_request()
        
    def acknowledge_requests(self, vars=None):
        """
        Wait for the results of the non-blocking requests that are
        sent to set the experiment variables. 
        
        Input: 
            vars (optional): list of all variables, for which the
                any outstanding requests should be acknowledge. If vars
                is None, all variables will be checked (default: None).
        Output: 
            results: dictionary of the results returned by the 
                acknowledge_request methods with variable names
                as the keys.
        """
        if vars is None:
            vars = self._vars
        else:
            if isinstance(vars, str):
                vars = [vars]
            elif isinstance(vars, list):
                pass
            else:
                raise ExperimentDefinitionError('Argument in ' +
                        'acknowledge_requests method should be ' +
                        'a variable name or a list containing ' +
                        'variable names.')
        results = {}
        for var in vars:
            results[var] = self.acknowledge_request(var)
        return results

    ###DATA SAVING METHODS##############################################
    def _txt_save(self, data):
        """
        Save the data in a human-readable text data file. The method
        saves a header containing all experiment variables, then sweep
        variables, and then the variables in the data dictionary.
        """
        text_dir = os.path.join(self._save_path, 'TextData')
        
        if not os.path.exists(text_dir):
            try:
                os.makedirs(text_dir)
            except:
                raise Exception('Could not create data save path for ' +
                        'the experiment data! Is AFS on?')
        
        # Which contents are files?
        only_files = [f for f in os.listdir(text_dir)
                     if os.path.isfile(os.path.join(text_dir, f))]
        expt_name = self.information['Experiment Name'].replace(" ", "_")
        # Which files start off with 'ExperimentName_'?
        files = [f.split('.')[0] for f in only_files 
                 if f[:len(expt_name) + 1] == expt_name + '_']
        # Get the file numbers and the increment, or create the first 
        # file if none in the folder.
        nums = [int(f[-3:]) for f in files if f[-3:].isdigit()]
        if not nums:
            num = '000'
            fname = expt_name + '_' + num + '.txt'
        else:
            num = ("%03d" % (max(nums) + 1,))
            fname = expt_name + '_' + num + '.txt'
        
        # Create the file path.
        file_path = os.path.join(text_dir, fname)
        
        # Build a header for the file.
        h = ['Format Version: 0.1', expt_name, time.asctime()]
        h.append('====Experiment Parameters====')
        # Save only the variables that have been actually used.
        # Do not save the sweep variables here.
        for var in self._vars:
            if (var not in data and 'Save' in self._vars[var] and 
                self._vars[var]['Save'] and 'Value' in self._vars[var]):
                h.append(var + ': ' +
                        self.val2str(self._vars[var]['Value'], True))

        if 'Comments' in self.information:
            h.append('Comments: ' + self.information['Comments'])
        
        h.append('====Sweep Variables====')
        with file(file_path, 'w') as outfile:
            for k in h:
                outfile.write(k + '\n')
            # This is to avoid duplicative saving of the sweep variables
            # when a parallel scan was run with the same variable name.
            for key in data:
                if self._is_indep(data[key]):
                    outfile.write("Name: '" + key + "'\n")
                    outfile.write('Type: independent\n')
                    self._write_var(outfile, data[key]['Value'])

            outfile.write('====Data Variables====\n');
            for key in data:
                if self._is_dep(data[key]):
                    outfile.write("Name: '" + key + "'\n")
                    outfile.write('Type: dependent\n')
                    if 'Dependencies' in data[key] and data[key]['Dependencies']:
                        outfile.write('Dependencies: ' +
                                str(data[key]['Dependencies'])[1:-1] + '\n')
                    if 'Distribution' in data[key] and data[key]['Distribution']:
                        outfile.write('Distribution: ' +
                                data[key]['Distribution'] + '\n')
                    self._write_var(outfile, data[key]['Value'])
            
    def _write_var(self, outfile, vals):
        """
        Write variable values into a text file, starting with the
        data size.
        
        Inputs:
            outfile: file handle to write the data into.
            vals: variable values.
        Output:
            None.
        """
        units = self.get_units(vals)
        if units != '':
            outfile.write('Units: ' + units + '\n')
        outfile.write('Size: ' + str(list(np.shape(vals)))[1:-1] + '\n')
        self._ndarray_txt_save(outfile, self.strip_units(vals))

    def _ndarray_txt_save(self, outfile, array):
        """
        Write numpy.ndarray into a text file.
        
        Inputs:
            outfile: file handle to write the data into.
            array: numpy.ndarray to append to the end of a specified 
                file.
        Output:
            None.
        """
        if array.ndim == 1 or array.ndim == 2:
            if (array[np.isfinite(array)].size and
                    (array == array.astype(int)).all()):
                format = '%d'
            else:
                format = '%-7.6f'
            np.savetxt(outfile, array, fmt=format, delimiter='\t')
        elif array.ndim > 2:
            for idx in range(array.shape[0]):
                self._ndarray_txt_save(outfile, array[idx])
                
    def _mat_save(self, data):
        """
        Save data as a .mat file using scipi.io. Data will be saved as
        a structure, with a substructure for the experiment and 
        electronics, and arrays for the data.
        """
        matlab_dir = os.path.join(self._save_path, 'MATLABData')
        
        if not os.path.exists(matlab_dir):
            try:
                os.makedirs(matlab_dir)
            except:
                raise Exception('Could not create experiment MATLAB ' + 
                'data save path! Is AFS on?')
        
        # Which contents are files?
        only_files = [f for f in os.listdir(matlab_dir)
                     if os.path.isfile(os.path.join(matlab_dir, f))]
        expt_name = self.information['Experiment Name'].replace(" ", "_")
        # Which files start off with 'ExperimentName_'?
        files = [f.split('.')[0] for f in only_files
                 if f[:len(expt_name) + 1] == expt_name + '_']
        
        # Get file numbers and increment, or create first file if none
        # in the folder.
        nums = [int(f[-3:]) for f in files if f[-3:].isdigit()]
        if nums==[]:
            num = '000'
            fname = expt_name + '_' + num + '.mat'
        else:
            num = ("%03d" % (max(nums) + 1,))
            fname = expt_name + '_' + num + '.mat'
        
        file_path = os.path.join(matlab_dir,fname)
        
        # Convert variable names to a MATLAB-friendly format.
        # Save the variables that have been actually used.
        # Do not save here the experiment variables that are sweep 
        # variables.
        matVars = {}
        matUnits = {}
        for var in self._vars:
            if (var not in data and 'Save' in self._vars[var] and 
                    self._vars[var]['Save'] and 'Value' in self._vars[var]):
                matVars[var.replace(" ", "_")] = self.strip_units(var)
                matUnits[var.replace(" ", "_")] = self.get_units(var)
        # Create a data dictionary.
        matData = {}
        for key in data:
            if 'Value' in data[key]:
                matData[key.replace(" ", "_")] = self.strip_units(data[key]['Value'])

        # Save the information about the data units and the expected distributions.
        matDatadistr = {}
        matDataDepend = {}
        for key in data:
            if 'Value' in data[key]:
                matUnits[key.replace(" ", "_")] = self.get_units(data[key]['Value'])
                if 'Distribution' in data[key]:
                    matDatadistr[key.replace(" ", "_")] = data[key]['Distribution']
                if 'Dependencies' in data[key]:
                    matDataDepend[key.replace(" ", "_")] = str(data[key]['Dependencies']).replace(", ", ",").replace(" ", "_")
        
        # Create dictionary that will be saved to a .mat file.
        saveDict = {'Time': time.asctime(),
                    'Name': expt_name + '_' + num,
                    'ExptVars': matVars,
                    'Data': matData,
                    'Comments': self.information['Comments']}

        if matUnits:
            saveDict['Units'] = matUnits
        if matDatadistr:
            saveDict['Distr'] = matDatadistr
        if matDataDepend:
            saveDict['Depend'] = matDataDepend

        sio.savemat(file_path, {saveDict['Name']: saveDict})                

    def _save_data(self, data):
        """
        Save the data in the text (.txt) and the MATLAB (.mat) file
        formats.
        
        Input:
            data: data dictionary.
        Output:
            None.
        """ 
        # Remove unnecessary dimensions. This means that the independent
        # variables that contain only one value should also be removed 
        # from the dependency specifications.
        
        # Find independent variables with a single value.
        rm_vars = []
        for key in data:
            if (self._is_indep(data[key]) and
                    np.size(data[key]['Value']) == 1):
                self.add_var(key, data[key]['Value'][0])
                rm_vars.append(key)
        for key in rm_vars:
            data.pop(key)
        
        # Remove unnecessary dimensions that contain only one value.
        for key in data:
            if 'Value' in data[key]:
                if isinstance(data[key]['Value'], units.Value):
                    u = self.unit_factor(data[key]['Value'])
                    data[key]['Value'] = np.array([data[key]['Value']]) * u
                else:
                    data[key]['Value'] = np.squeeze(data[key]['Value'])
                    # Convert single numbers (zero-dimensional numpy arrays) 
                    # to one-dimensional numpy arrays.
                    if np.size(data[key]['Value']) == 1:
                        data[key]['Value'] = data[key]['Value'].reshape(1)
            if 'Dependencies' in data[key]:
                data[key]['Dependencies'] = [var for var 
                        in data[key]['Dependencies'] if var not in rm_vars]

        # Remove 'Repetition Index', 'Run Iteration', and 'Long
        # Repetition Index' independent  variables if they are not
        # actually used. This may happen when the data dimensions are
        # truncated.
        
        # Create a list containing all independent variable.
        indep_vars = []
        for key in data:
            if 'Value' in data[key] and 'Dependencies' in data[key]:
                for var in data[key]['Dependencies']:
                    indep_vars.append(var)

        # Remove unnecessary independent variables.
        for var in ['Repetition Index', 'Run Iteration',
                'Long Repetition Index']:
            if var not in indep_vars:
                data.pop(var, None)
            self._vars.pop(var, None)
        
        self._txt_save(data)
        self._mat_save(data)
        print('The data has been saved.')
    
    ###UTILITIES########################################################
    def get_units(self, v):
        """
        Return variable units.
        
        Input:
            v: value or name of the variable.
        Output:
            units: the variable units. 
        """
        if v is None:
            return ''
        
        if isinstance(v, units.Value):
            return str(units.Unit(v))
        
        if isinstance(v, list):
            if len(v) == 1 and isinstance(v[0], str):
                v = v[0]
            elif all([isinstance(val, np.ndarray) for val in v]):
                v = np.array(v)
            elif all([isinstance(val, units.Value) for val in v]):
                unit = list(set([units.Unit(val) for val in v]))
                if len(unit) == 1:
                    return str(units.Unit(unit[0]))
                else:
                    raise Exception("More than one physical unit is" +
                                    " found: " + str(unit) + ".")
            # Be careful: isinstance(0 * units.K, float) returns True...
            elif all([isinstance(val, (int, long, float, complex))
                    for val in v]):
                return ''
        
        if isinstance(v, np.ndarray):
            if all([isinstance(val, (np.ndarray, units.Value)) for val in v.flatten()]):
                unit = list(set([self.get_units(val) for val in v.flatten()]))
                if len(unit) == 1:
                    return str(units.Unit(unit[0]))
                else:
                    raise Exception("More than one physical unit is" +
                                    " found: " + str(unit) + ".")
            else:
                return ''
        
        if isinstance(v, str):
            self._check_var(v, check_interface=False)
            value = self._vars[v]['Value']
            if isinstance(value, units.Value):
                return str(units.Unit(value))
            else:
                return ''
        
        # Be careful: isinstance(1 * units.GHz, float) returns True...
        if isinstance(v, (int, long, float, complex)):
            return ''
        
        raise Exception("No units can be obtained for '" + str(v) + 
                        "' of type '" + str(type(v)) + "'.")
            
    def strip_units(self, v):
        """
        Return the variable value as a number without any units. 
        Use this method only when it is really necessary to do so.
        
        Input:
            v: value or name of the variable.
        Output:
            number: numerical value of the variable. 
        """
        if v is None:
            return None
        if isinstance(v, units.Value):
            return v[units.Unit(v)]
        if isinstance(v, (int, long, float, complex)):
            return v
        if isinstance(v, np.ndarray):
            shape = np.shape(v)
            v = v.flatten()
            stripped = np.empty(np.shape(v))
            for k in range(np.size(v)):
                stripped[k] = self.strip_units(v[k])
            return stripped.reshape(shape)
        if isinstance(v, list):
            return np.vectorize(self.strip_units)(np.array(v))
        if isinstance(v, str):
            self._check_var(v, check_interface=False)
            if isinstance(self._vars[v]['Value'], units.Value):
                return self._vars[v]['Value'][units.Unit(self._vars[v]['Value'])]
            else:
                return self._vars[v]['Value']
        raise Exception("Units could not be stripped from '" + str(v) + 
                "', which is of type '" + str(type(v)) + "'.")
                
    def unit_factor(self, values):
        """
        Determine the unit multiplier for an object that contains
        some data values.
        
        Input:
            values: value(s) to extract the unit from.
        Output:
            unit_multiplier: units.Unit object or 1 for dimensionless 
                numbers.
        """
        unit = units.Unit(self.get_units(values))
        if unit == units.Unit(''):
            return 1
        else:
            return unit
    
    def val2str(self, val, brackets=False):
        """
        Return variable units.
        
        Inputs:
            val: value of a physical quantity or a unitless number.
            brackets (optional): boolean that specifies whether the 
                variable units should be enclosed in brackets 
                (default: False).
        Output:
            string: string representing the physical quantity or
                a unitless number.
        """
        if isinstance(val, units.Value) and brackets:
            return (str(val[units.Unit(val)]) + ' [' + 
                    str(units.Unit(val)) + ']')
        else:
            return str(val)
            
    def _comb_strs(*args):
        """
        Combine strings and lists of strings and return a list of 
        strings with unique elements.
        
        Inputs:
            *args: strings and/or lists of strings.
        Output:
            list of strings: flatten list containing only unique 
                strings. 
        """
        vars = []
        for arg in args:
            if arg is not None:
                if isinstance(arg, str):
                    vars = vars + [arg]
                elif isinstance(arg, list):
                    if any([not isinstance(var, str) for var in arg]):
                        print('Warning: internal method _comb_strs' +
                        ' can only accept strings or lists of strings' +
                        ' as its arguments.')
                    else:
                        vars = vars + arg
        result = []
        for var in vars:
            if var not in result:
                result.append(var)

        return result

    ###EXPERIMENT CONTROL METHODS#######################################
    def value(self, var, value=None, output=False, soft=True):
        """
        Get or set a single variable in the experiment variables 
        dictionary. Useful if you want to run a single point experiment
        or over a few different values. 
        
        Input(s):
            var: name of the variable key in the experiment variables
                dictionary.
            value (optional): if value is None or omitted, the function
                returns the value of variable var (default: None). 
                Otherwise, the function sets variable to the specified
                value if variable var could be found among the
                experiment variables. The method raises an exception if
                the variable is not found.
            output (optional): print a message showing a new value.
            soft (optional): throw an exception if the variable doesn't
                exist or doesn't have a value asssigned to it (default:
                True).
        Output:
            value: current value of the specified variable or None.
        """
        if value is None:
            if var in self._vars and 'Value' in self._vars[var]:
                return self._vars[var]['Value']
            elif soft:
                return None
            else:
                raise Exception("Variable '" + str(var) + "' does not" +
                        " exist or does not have a value assigned to it.")
        else:
            self._check_var(var, check_value=False, check_interface=False)
            if 'Value' in self._vars[var]:
                if ((isinstance(value, units.Value) !=
                     isinstance(self._vars[var]['Value'], units.Value))
                     or
                    (isinstance(value, units.Value) and not
                     value.isCompatible(units.Unit(self._vars[var]['Value'])))):
                    raise Exception("An attempt to change the '" +
                            str(var) + "' units is detected. Check " +
                            "variable value reassignments.")
            self._vars[var]['Value'] = value
            self._vars[var]['Save'] = True
            if output:
                print("Experiment variable '" + var + 
                "' is set to " + self.val2str(value) + ".")
            return value

    ###EXPERIMENT RUN FUNCTIONS#########################################
    def init_expt(self):
        """
        This method is called before any load_once and run_once methods
        by the sweep method and could be used to reduce the overhead
        caused by the need to preset the device resources and/or do any 
        initializations.
        
        Input: 
            None.
        Output:
            None.
        """
        pass
        
    def exit_expt(self):
        """
        This method is called after all load_once and run_once methods
        by the sweep method and could be used to reduce the overhead
        caused by the need to reset the deviceresources and/or do any 
        data post-processing.
        
        Input: 
            None.
        Output:
            None.
        """
        pass
    
    def load_once(self):
        """
        This method is called before run_once by the sweep method and
        could be used to reduce the overhead
        caused by the need to set the FPGA boards and other devices
        when multiple runs of the same experiment have to be executed.
        
        Input: 
            None.
        Output:
            None.
        """
        pass

    def run_once(self):
        """
        Basic run method. This method should be modified in each 
        inherited class. The method is called by the sweep method. 
        In the inherited classes run_once should define the logic
        and basic data structure of a single experiment. Method
        load_once could be used to reduce the overhead
        caused by the need to set the FPGA boards and other devices
        when multiple runs of the same experiment have to be executed.
        
        Input: 
            None.
        Output:
            None.
        """
        pass

    def run_n_times(self, runs):
        """
        This method is similar to run_once with the exception that 
        the experiment defined by run_once method will be called
        a specified number of times and the results of all runs will be
        averaged.
        
        Input: 
            runs: number of independent runs of the experiment.
        """
        self.add_var('Runs', runs)
        if self._standard_output:
            sys.stdout.write('Progress:   0.0%\r')
        
        for idx in range(runs):
            run_data = self.run_once()
            if self._standard_output:
                sys.stdout.write('Progress: %5.1f%%\r' %(100. * (idx + 1) / runs))
            
            reps = self.value('Reps')
            
            if idx == 0 and self._sweep_pts_acquired == 0:
                self._run_n_data = self._process_data(run_data)
                n_data = self._run_n_data
                self._run_n_data_deps = [key for key in run_data
                        if self._is_dep(n_data[key])]

                self._run_n_data_flat = []
                if reps is not None and reps > 1:
                    for key in run_data:
                        if (self._is_dep(n_data[key]) and
                                len(n_data[key]['Dependencies']) > 0 and
                                n_data[key]['Dependencies'][0] == \
                                'Repetition Index'):
                            n_data[key]['Dependencies'][0] = \
                                'Long Repetition Index'
                            self._run_n_data_flat.append(key)
                            self._run_n_data_deps.remove(key)

                for key in self._run_n_data_deps:
                    entry_shape = (runs,) + np.shape(run_data[key]['Value'])
                    if self.get_units(run_data[key]['Value']) != '':
                        n_data[key]['Value'] = np.empty(entry_shape,
                                            dtype=units.Value)
                    else:
                        n_data[key]['Value'] = np.empty(entry_shape)
                n_data['Run Iteration'] = {'Value': 
                        np.linspace(1, runs, runs), 'Type': 'Independent'}

                for key in self._run_n_data_flat:
                    entry_shape = (runs * reps,) + np.shape(run_data[key]['Value'])[1:]
                    if self.get_units(run_data[key]['Value']) != '':
                        n_data[key]['Value'] = np.empty(entry_shape,
                                            dtype=units.Value)
                    else:
                        n_data[key]['Value'] = np.empty(entry_shape)
                    n_data['Long Repetition Index'] = {'Value': 
                            np.linspace(1, runs * reps, runs * reps),
                            'Type': 'Independent'}
            else:
                n_data = self._run_n_data

            for key in self._run_n_data_deps:
                n_data[key]['Value'][idx] = run_data[key]['Value']
                
            for key in self._run_n_data_flat:
                n_data[key]['Value'][reps*idx:reps*(idx+1)] = \
                    run_data[key]['Value']
            
            self._listen_to_keyboard()
            if self._sweep_status == 'abort':
                # Delete unrun 'Run Iterations'.
                n_data['Run Iteration']['Value'] = np.delete(
                        n_data['Run Iteration']['Value'],
                        np.s_[idx+1:], None)
                if self._run_n_data_flat:
                    n_data['Long Repetition Index']['Value'] = np.delete(
                        n_data['Long Repetition Index']['Value'],
                        np.s_[reps*(idx+1):], None)
                # Delete unfilled data points.
                for key in self._run_n_data_deps:
                    n_data[key]['Value'] = np.delete(n_data[key]['Value'],
                            np.s_[idx+1:], 0)
                for key in self._run_n_data_flat:
                    n_data[key]['Value'] = np.delete(n_data[key]['Value'],
                            np.s_[reps*(idx+1):], 0)
                break
 
        self.average_data()
        return self._avg_data

    def average_data(self):
        """
        Average the data acquired by method run_n_times.
        """
        data = self._data
        if self._sweep_pts_acquired == 0:
            self._avg_data = {key: data[key].copy() for key in data}

        avg = self._avg_data
        for key in data:
            if self._is_dep(data[key]):
                vals = self.strip_units(data[key]['Value'])
                u = self.unit_factor(data[key]['Value'])
                avg[key]['Distribution'] = 'normal'
                avg[key]['Value'] = np.mean(vals, axis=0) * u
                avg[key + ' Std Dev'] = data[key].copy()
                avg[key + ' Std Dev']['Distribution'] = ''
                avg[key + ' Std Dev']['Value'] = np.std(vals, axis=0) * u

    def _is_indep(self, data_subdict):
        """
        Check wheather a given data subdictionary corresponds to
        an independent variable.
        
        Input:
            data_subdict: data subdictionary, e.g. if data is
                a dictionary that contains all data and var is
                an independent variable then data_subdict is data[var].
        Output:
            bool: True or False.
        """
        if ('Value' in data_subdict and
             'Type' in data_subdict and
                       data_subdict['Type'] == 'Independent'):
            return True
        else:
            return False
            
    def _is_dep(self, data_subdict):
        """
        Check whether a given data subdictionary corresponds to
        a dependent variable.
        
        Input:
            data_subdict: data subdictionary, e.g. if data is
                a dictionary that contains all data and var is
                a dependent variable then data_subdict is data[var].
        Output:
            bool: True or False.
        """
        if ('Value' in data_subdict and
             'Type' in data_subdict and
                       data_subdict['Type'] == 'Dependent'):
            return True
        else:
            return False
        
    def _process_data(self, raw_data):
        """
        Check that the data dictionary returned by a single run of
        an experiment meets the minimum consistency criteria. Create
        the field entries that were omitted.
        
        Input:
            raw_data: data dictionary returned, for example, by run_once
                method.
        Output:
            data: processed data dictionary potentially supplemented 
                with some extra fields.
        """
        if not isinstance(raw_data, dict):
            raise DataError("Data should be given in a dictionary " +
                    "format.")

        if any([not isinstance(raw_data[entry], dict) for entry in raw_data]):
            raise DataError("Each element of the data dictionary " +
                    "should be dictionary by itself. Use key 'Value' " +
                    "inside the subdictionaries " +
                    "to specify actual data values.")  
            
        # Assign 'Type' to be 'Independent' if the variable
        # is listed in any dependency specifications.
        data = {}
        for key in raw_data:        # Make a deep copy.
            data[key] = raw_data[key].copy()
        for key in data:
            if 'Dependencies' in data[key]:
                if isinstance(data[key]['Dependencies'], str):
                    data[key]['Dependencies'] = [data[key]['Dependencies']]
                for dep in data[key]['Dependencies']:
                    if dep not in data:
                        raise DataError("Independent data variable '" +
                        str(dep) + "' as an independent variable for '" +
                        str(key) + "' is not found in the data dictionary.")
                    if 'Value' not in data[dep]:
                        raise DataError("Independent data variable '" + 
                        str(dep) + "' does not have 'Value' entry.")
                    if 'Type' in dep:
                        if 'Type' != 'Independent':
                            raise DataError("Conflicted dependencies " +
                            "are specified in the '" + str(key) + 
                            "' and '" + str(dep) + "' data entries.")
                    else:
                        data[dep]['Type'] = 'Independent'
        
        # Check that data variables containing more than one element
        # have their dependencies properly specified.
        for key in data:
            if 'Value' in data[key] and ('Type' not in data[key] or
                    data[key]['Type'] == 'Dependent'):
                if (isinstance(data[key]['Value'], np.ndarray)
                    and data[key]['Value'].size > 1):
                    if 'Dependencies' not in data[key]:
                        raise DataError("Data variable '" + str(key) + 
                        "' is given without a dependency specification.")
                    if isinstance(data[key]['Dependencies'], list):
                        if not all([isinstance(name, str) for name in 
                                data[key]['Dependencies']]):
                            raise DataError("Data variable '" + str(key) + 
                            "' dependencies should be specified as a string" +
                            " or a list of strings.")
                    expected_shape = ()
                    for dep in data[key]['Dependencies']:
                        expected_shape = expected_shape + np.shape(data[dep]['Value'])
                    if np.shape(data[key]['Value']) != expected_shape:
                        raise DataError("Data variable '" + str(key) + 
                        "' size " + str(np.shape(data[key]['Value'])) +
                        " differs from the sizes of the independent " +
                        "variables " + str(expected_shape) + ".")
                elif 'Dependencies' not in data[key]:
                    data[key]['Dependencies'] = []
                elif len(data[key]['Dependencies']) > 0:
                    raise DataError("Data variable '" + str(key) + 
                        "' is not a numpy array containing more than " +
                        "one element and, thus, should not have any " +
                        "dependencies explicitly specified.")
        
        # Ensure that the types are assigned correctly.
        for key in data:
            if 'Type' not in data[key]:
                data[key]['Type'] = 'Dependent'
            elif data[key]['Type'] not in ['Dependent', 'Independent']:
                raise DataError("The data entry 'Type' should " +
                "be either 'Dependent' or 'Independent'.")
        for key in data:
            if (data[key]['Type'] == 'Dependent' and
                'Dependencies' not in data[key]):
                raise DataError("Data variable '" + str(key) + 
                        "' does not have 'Dependencies' entry specified.")
        
        # Add independent variables to the list of experiment variables.
        for key in data:
            if 'Type' in data[key] and data[key]['Type'] == 'Independent':
                if 'Value' in data[key]:
                    if isinstance(data[key]['Value'], np.ndarray):
                        value = data[key]['Value']
                        for _ in np.shape(data[key]['Value']):
                            value = value[-1]
                    else:
                        value = data[key]['Value']
                else:
                    value = None
                self.add_var(key, value)
        
        return data
            
    def _sweep(self, names, values, 
               print_expt_vars=None, print_data_vars=None,
               plot_data_vars=None, max_data_dim=2, runs=1):
        """
        Run an N-dimensional sweep over a given set of variables. In the
        most general case names and values should be a list of lists of 
        strings and 1D numpy arrays of the same structure. The 
        experiment and data variables that should printed and/or plotted
        should be specified when necessary.
        
        Inputs:
            names: names of the variable over which to sweep.
            values: values over which the sweep the variables.
            save: the data will be saved if this flag is True.
            print_data_vars (optional): data variables to print to the 
                standard output as a list of strings (or a string).
            plot_data_vars (optional): data variables to plot as a list
                of strings (or a string).
            max_data_dim (optional): limit on the maximum dimension of 
                the data array. This prevents saving unnecessary large
                data sets when it is not required.
            runs (optional): number of experiment runs at each point, 
                the output data will be averaged.
        Output:
            None.
        """
        
        if len(names[0]) == 1:      # Run a 1D sweep.
            for idx in range(values[0][0].size):
                for p_idx in range(len(names)):
                    self.value(names[p_idx][0], values[p_idx][0][idx], False)

                # Load the board settings and set the instruments.
                self.load_once()
                if runs == 1:
                    # Run the experiment once.
                    run_data = self.run_once()
                else:
                    # Run the same experiment multiple times.
                    run_data = self.run_n_times(runs)
                
                self._sweep_pts_acquired = self._sweep_pts_acquired + 1 

                if idx == 0:
                    if self._sweep_pts_acquired == 1:
                        # Initialize the data dictionary.
                        # Check that the dictionary is properly defined.
                        self._1d_data = self._process_data(run_data)
                        
                        # Dependent variables, values of which should
                        # be kept in memory and, potentially, saved.
                        self._1d_data_deps = []
                        
                        # Preallocate the memory resources.
                        for key in self._1d_data:
                            if self._is_dep(self._1d_data[key]):
                                entry_shape = (np.shape(values[0][0]) + 
                                        np.shape(self._1d_data[key]['Value']))
                                if len(entry_shape) <= max_data_dim:
                                    if self.get_units(self._1d_data[key]['Value']) != '':
                                        self._1d_data[key]['Value'] = \
                                                np.empty(entry_shape,
                                                dtype=units.Value)
                                    else:
                                        self._1d_data[key]['Value'] = \
                                                np.empty(entry_shape)
                                    self._1d_data_deps.append(key)
                                else:
                                    self._1d_data[key].pop('Value')

                    # Make a list of the sweep variables that should be 
                    # printed to the standard output.
                    for var in self._comb_strs(print_expt_vars):
                        if var not in self._vars:
                            print("Warning: variable '" + str(var) + 
                            "' is not found among the experiment " +
                            "variables: " +
                            str([var for var in self._vars]) + 
                            ". The value of this variable will not " +
                            "be printed.")
                    print_expt_vars = [var for var in
                            self._comb_strs(print_expt_vars)
                                if var in self._vars]

                    # Make a list of the data variables that should be
                    # printed to the standard output.
                    if print_data_vars is None:
                        print_data_vars = [var for var in run_data
                                if np.size(run_data[var]['Value']) == 1]
                    else:
                        for var in self._comb_strs(print_data_vars):
                            if var not in run_data:
                                print("Warning: variable '" + str(var) +
                                "' is not found among the data " + 
                                "dictionary keys: " + str(run_data.keys()) + 
                                ". This data will not be printed.")
                        print_data_vars = [var for var
                                in self._comb_strs(print_data_vars)
                                if var in run_data and
                                np.size(run_data[var]) == 1]

                # Add the newly acquired data to the data set.
                for key in self._1d_data_deps:
                    self._1d_data[key]['Value'][idx] = run_data[key]['Value']

                if idx == 0:
                    plot_data_vars = self._init_1d_plot(names, values, 
                            self._1d_data, plot_data_vars)
                # Print experiment and data variables to the standard
                # output.
                if self._standard_output and self._sweep_status!= 'abort':
                    for var in print_expt_vars:
                        print(var + ' = ' + 
                                self.val2str(self._vars[var]['Value']))
                    for var in print_data_vars:
                        print(var + ' = ' + 
                                self.val2str(run_data[var]['Value']))
                
                # Update the plot if anything is being plotted.
                if plot_data_vars is not None:
                    self._update_1d_plot(names, values, self._1d_data,
                                         plot_data_vars, idx)

                if self._sweep_status == '':
                    # Check whether any key is pressed.
                    self._listen_to_keyboard()
                if self._sweep_msg != '':
                    print(self._sweep_msg)
                    self._sweep_msg = ''
                if self._sweep_status == 'abort':
                    break
                elif self._sweep_status == 'abort-and-save':
                    # The scan has been aborted.
                    # Delete unused sweep variable values.
                    for p_idx in range(len(values)):
                        values[p_idx] = [np.delete(values[p_idx][0],
                                np.s_[idx+1:], None)]
                    # Delete unfilled data points since the data 
                    # has been previously initialize with np.empty.
                    for key in self._1d_data_deps:
                        self._1d_data[key]['Value'] = \
                                np.delete(self._1d_data[key]['Value'],
                                np.s_[idx+1:], 0)
                    break

            return self._1d_data, values
        else:
            # Recurrent implementation of the multidimensional sweeps.
            run_names = []
            run_vals = []
            for p_idx in range(len(names)):
                run_names.append(names[p_idx][1:])
                run_vals.append(values[p_idx][1:])
                
            for idx0 in range(values[0][0].size):
                # Assign new values for the sweep variables.
                for p_idx in range(len(names)):
                    self.value(names[p_idx][0], values[p_idx][0][idx0], False)
                run_data, vals = self._sweep(run_names, run_vals, 
                        print_expt_vars, print_data_vars, 
                        plot_data_vars, max_data_dim-1, runs)
                
                if idx0 == 0:
                    # Initialize the data dictionary.
                    data = {}
                    for key in run_data:    # Make a deep copy.
                        data[key] = run_data[key].copy()
                    
                    data_deps = []
                    for key in data: 
                        if self._is_dep(data[key]):
                            entry_shape = (np.shape(values[0][0]) + 
                                    np.shape(data[key]['Value']))
                            if len(entry_shape) <= max_data_dim:
                                if self.get_units(data[key]['Value']) != '':
                                    data[key]['Value'] = np.empty(entry_shape,
                                            dtype=units.Value)
                                else:
                                    data[key]['Value'] = np.empty(entry_shape)
                                data_deps.append(key)
                            else:
                                data[key].pop('Value')

                if self._sweep_status == 'abort-and-save':
                    if idx0 == 0:
                        # The experiment was aborted during 
                        # the acquisition of the first slice.
                        for p_idx in range(len(values)):
                            values[p_idx] = ([np.array([values[p_idx][0][0]]) *
                                    self.unit_factor(values[p_idx][0][0])] +
                                    vals[p_idx])
                        data = run_data
                    elif np.shape(vals[0][0]) == np.shape(values[0][1]):
                        # A full data slice has been acquired.
                        for p_idx in range(len(values)):
                            values[p_idx] = [np.delete(values[p_idx][0], 
                                    np.s_[idx0+1:], None)] + values[p_idx][1:]
                        for key in data_deps:
                            data[key]['Value'][idx0] = run_data[key]['Value']
                            data[key]['Value'] = np.delete(data[key]['Value'], 
                                    np.s_[idx0+1:], 0)
                    else:
                        # At least one slice has been acquired but the 
                        # current slice hasn't been finished.
                        for p_idx in range(len(values)):
                            values[p_idx] = [np.delete(values[p_idx][0],
                                    np.s_[idx0:], None)] + values[p_idx][1:]
                        for key in data_deps:
                            data[key]['Value'] = np.delete(data[key]['Value'], 
                                    np.s_[idx0:], 0)
                    break
                elif self._sweep_status == 'abort':
                    break
                
                # Add the newly acquired data to the data set.
                for key in data_deps:
                    data[key]['Value'][idx0] = run_data[key]['Value']

            return data, values

    def sweep(self, names, values, save=False, 
            print_data=None, plot_data=None, 
            dependencies=None, max_data_dim=2, runs=1):
        """
        Run an N-dimensional sweep over a given set of variables, 
        defined by the keys in the experiment variable dictionary.
        In the most general case names and values should be list of
        lists of strings and 1D numpy arrays. The data will be saved
        with some non-public methods if save flag is True.
        The experiment and data variables that should printed and/or
        plotted could be specified when necessary.
        
        Inputs:
            names: names of the variable over which to sweep.
            values: values over which the sweep the variables.
            save: the data will be saved if this flag is True.
            print_data (optional): data variables to print to 
                the standard output as a list of strings (or a string).
            plot_data (optional): data variables to plot as a list
                of strings (or a string).
            dependencies (optional): dependency specifications.
            max_data_dim (optional): limit on the maximum dimension
                of the data array. This prevents saving unnecessary
                large data sets when it is not required.
            runs (optional): number of the experiment runs at each
                point, the output data will be averaged (using 
                _average_data method).
        Output:
            None.
        
        Examples:
            run.sweep([['RF Frequency'], ['RF Frequency']],
                      [[np.array([1e9, 2e9, 3e9] * GHz)], 
                       [np.array([1e9, 2e9, 3e9] * GHz)]], 
                      save=True, print_data=['Pa', 'Pb'],
                      plot_data=['Pa', 'Pb'], 
                      dependencies=[['Pa'], ['Pb']])
                 
            run.sweep([['x'], ['y']], [[xval], [yval]], save=True,
                      print_data=['Result 1', 'Result 2'], 
                      plot_data=['Result 1', 'Result 2'])
        """
        if not hasattr(self, '_vars'):
            raise Exception('Experiment resources and variables ' +
            'should be set prior attempting any sweep measurements.')

        excpt_msg = ('The first argument in sweep method should be'
        ' either a string (for a 1D simple scan), a list of strings'
        ' (for a multidimensional scan), or a list of lists of strings'
        ' (for a parallel any-dimensional scan).')
        
        if isinstance(names, str):
            # Check that the variable is properly defined.
            self._check_var(names, check_interface=False)
            # If there is only one sweep variable defined as a string,
            # convert it to a list of lists for internal code consistency.
            names = [[names]]
        elif isinstance(names, list):
            # Check that there are more than zero elements and confirm 
            # that they are strings.
            if len(names) > 0 and all([isinstance(n, str) for n in names]):
                for name in names:
                    self._check_var(name, check_interface=False)
                # Check that the variables are unique.
                if len(names) > len(set(names)):
                    raise SweepError('Sweep method was called with ' +
                    'repeated variable names. The data might be hard ' +
                    'to interpret.')
                # If there are no independent parallel scans, convert 
                # the list to a list of lists for internal code
                # consistency.
                names = [names]
            elif (len(names) > 0 and all([isinstance(ln, list)
                    for ln in names]) and len(names[0]) > 0):
                # Do the following checks if a list of lists of strings 
                # is actually specified.
                for name_list in names:
                    # Check that the nested lists have the same length.
                    if len(name_list) != len(names[0]):
                        raise SweepError('The length of name lists ' +
                        'should be the same for any parrallel scans.') 
                    # Check that the variables are properly defined.
                    for name in name_list:
                        self._check_var(name, check_interface=False)
                    # Check that the variables are unique.
                    if len(name_list) > len(set(name_list)):
                        raise SweepError('Sweep method was called ' +
                        'with repeated variable names.')
                # Check that there are no repeated variable names along
                # different scan axes.
                for list_idx1, name_list1 in enumerate(names):
                    for name_idx1, name1 in enumerate(name_list1):
                        for name_list2 in names[list_idx1+1:]:
                            for name_idx2, name2 in enumerate(name_list2):
                                # Repeated variable names along the same 
                                # axis are allowed!
                                if name_idx1 != name_idx2 and name1 == name2:
                                    raise SweepError('Sweep method was' + 
                                    ' called with repeated variable' +
                                    ' names along differenent scan axes.')
            else:
                raise SweepError(excpt_msg)
        else:
            raise SweepError(excpt_msg)
        
        excpt_msg = ('The second argument in sweep method should be'
        ' either a 1D numpy array (for a 1D simple scan), a list of 1D'
        ' numpy arrays (for a multidimensional scan) or a list of lists'
        ' of 1D numpy arrays (for a parallel any-dinesional scan). For'
        ' parallel scans, the length of the 1D numpy arrays should be'
        ' equal along the same scan axis. Each numpy array should'
        ' contain at least one number.')
        if isinstance(values, np.ndarray):
            # Check the number of dimensions.
            if np.ndim(values) != 1:
                raise SweepError(excpt_msg)
            # If there is only one sweep variable defined as a 1D numpy 
            # array, convert it to a list of lists for internal code
            # consistency.
            values = [[values]]
        elif isinstance(values, list):
            if len(values) > 0 and all([isinstance(v, np.ndarray)
                                        for v in values]):
                # Check the number of the dimensions.
                for value in values:
                    if np.ndim(value) != 1:
                        raise SweepError(excpt_msg)
                # If there are no independent parallel scans, convert
                # the list of 1D numpy arrays to a list of lists for 
                # internal code consistency.
                values = [values]
            elif (len(values) > 0 and all([isinstance(v, list)
                    for v in values]) and len(values[0]) > 0):
                # Do the following checks if a list of lists of 1D numpy
                # arrays is actually specified.
                for value_list in values:
                    # Check that the nested lists have the same length.
                    if len(value_list) != len(values[0]):
                        raise SweepError('The length of 1D numpy ' +
                                'ndarray lists should be the same ' +
                                'for each parrallel scan.')
                    # Check that all sub-nested elements are non-empty
                    # 1D numpy arrays.
                    for value in value_list:
                        if (not isinstance(value, np.ndarray) or 
                            np.ndim(value) != 1 or np.size(value) == 0):
                            raise SweepError(excpt_msg)
                # Check that numpy arrays have the same length along
                # the same scan axis.
                for list_idx, value_list in enumerate(values):
                    for value_idx, value in enumerate(value_list):
                        if (len(values[0]) <= value_idx or
                            (list_idx > 0 and
                             values[0][value_idx].size != value.size)):
                            raise SweepError(excpt_msg)
            else:
                raise SweepError(excpt_msg)
        else:
            raise SweepError(excpt_msg)
        
        # Check that the sweep variables (names) and values can be
        # directly mapped to each other.
        if len(names) != len(values):
            raise SweepError('The sweep variable names could not be' +
                ' unambigiously matched to the specified values.')
        for list_idx, name_list in enumerate(names):
            if len(name_list) != len(values[list_idx]):
                raise SweepError('The sweep variable names could ' +
                        'not be unambigiously matched to the ' +
                        'specified values.')
                    
        # Check the dimension of the scan.
        if len(names[0]) > max_data_dim:
            raise SweepError("Maximum data dimesion is set to " +
                    str(max_data_dim) + " while the dimension of the "
                    "scan is " + str(len(names[0])) + ". Set key " +
                    "'max_data_dim' to a value that is not smaller "
                    "than the dimension of the scan.")
        
        # Prevent any attempts to set any sweep variable twice to some
        # different values. If this is allowed, it would be really hard
        # to interpret the data.
        for list_idx1, name_list1 in enumerate(names):
            for name_idx1, name1 in enumerate(name_list1):
                for list_idx2, name_list2 in enumerate(names):
                    if (list_idx2 > list_idx1 and
                        name1 == name_list2[name_idx1] and 
                        (values[list_idx1][name_idx1] != 
                         values[list_idx2][name_idx1]).any()):
                        raise SweepError('Sweep method was called with' +
                        ' repeated variable names along some scan axis' +
                        ' but the 1D numpy value arrays corresponding' +
                        ' to these variables are not the same.')
        
        excpt_msg = ('Optional parameter dependencies in sweep ' +
        'method should be either a string (for a 1D simple scan), '
        'a list of strings (for a single multidimensional scan), or a '+
        'list of string lists (for parallel any-dimensional scans).')
        # If any dependencies are specified, check the following.
        if dependencies is not None:
            # If there is only one dependency is specified, convert it
            # to a list of lists for internal code consistency.
            if isinstance(dependencies, str):
                dependencies = [[dependencies]]
            elif isinstance(dependencies, list):
                # If only a simple list is specified, convert this list
                # of strings to a list of lists for internal code 
                # consistency.
                if len(dependencies) > 0 and all([isinstance(dep, str)
                                         for dep in dependencies]):
                    dependencies = [list(set(dependencies))]
                elif len(dependencies) > 0 and all([isinstance(dep, list)
                                           for dep in dependencies]):
                    for list_idx, dep_list in enumerate(dependencies):
                        # Check that all sub-nested elements are strings.
                        if any([not isinstance(dep, str) for dep in dep_list]):
                            raise SweepError(excpt_msg)
                        dependencies[list_idx] = list(set(dep_list))
                    for list_idx, dep_list1 in enumerate(dependencies):
                        # Check that the same data variable does not 
                        # have any duplicated dependency specifications.
                        for dep1 in dep_list1:
                            for dep_list2 in dependencies[list_idx+1:]:
                                for dep2 in dep_list2:
                                    if dep1 == dep2:
                                        raise SweepError('Sweep method' +
                                        ' was called with conflicted' +
                                        ' dependency specifications.')
                else:
                    raise SweepError(excpt_msg)
            else:
                raise SweepError(excpt_msg)
            # Check that the sweep variables (names) and dependencies 
            # can be directly mapped to each other.
            if len(dependencies) != len(names):
                raise SweepError('Dependency specifications could not' +
                        'be unambigiously matched to sweep variables.')

        # Save the current sweep variables values. This allows running
        # several sweeps in a sequence without worrying about the sweep
        # variables too much. For example, one can start a frequency
        # sweep followed by a bias voltage sweep. Instead of leaving the
        # frequency at the last value of the sweep range, we want it to
        # return to the initial value. Usually, the initial value is
        # an important special case (could be, for instance, some
        # resonance frequency) and without the code below it would be
        # required to reset the value manually, which could be annoying.
        unique_names = set([name for sublist in names for name in sublist])
        prev_vals = {name: self.value(name) for name in unique_names}
                
        self._sweep_status= ''          # E.g. 'abort' or 'abort-and-save'.
        self._sweep_msg = ''
        self._sweep_dimension = len(names[0])   # Dimension of the sweep.
        self._sweep_start_time = time.time()    # Start time for the finish
                                                # time estimation.
        self._sweep_number_of_pts = 1   # Total number of sweep points.
        for val in values[0]:
            self._sweep_number_of_pts = len(val) * self._sweep_number_of_pts
        self._sweep_pts_acquired = 0    # Number of the acquired data points.

        print('\nStarting a ' + str(self._sweep_dimension) + 
              'D data sweep...\n' + 
              '\n\t[ESC]:\tAbort the run.' + 
              '\n\t[S]:\tAbort the run but [s]ave the data.' +
              '\n\t[T]:\tEstimate the finish [t]ime for the current scan.' +
              '\n\t[O]:\tTurn [o]n/[o]ff data printing to the standard [o]utput.' + 
              '\n\t[X]:\tSecret option.\n')
        
        print_expt_vars = []
        for name_list in names:
            print_expt_vars = print_expt_vars + name_list

        self.init_expt()
        
        data, values = self._sweep(names, values, print_expt_vars,
                print_data, plot_data, max_data_dim, runs)

        if ((save and self._sweep_status!= 'abort') or
            self._sweep_status == 'abort-and-save'):
            # If there is at least one point to save then...
            if values[0][0].size > 0:
                # Fully specify dependencies of
                # the data variables on the sweep variables.
                # This simplifies work-flow when several
                # independent measurements are done in parallel.
                if dependencies is None and len(names) == 1:
                    dependencies = [[key for key in data 
                            if self._is_dep(data[key])]]
                if dependencies is not None:
                    for list_idx, dep_list in enumerate(dependencies):
                        for dep in dep_list:
                            if dep in data:
                                data[dep]['Dependencies'] = \
                                        (names[list_idx] + 
                                        data[dep]['Dependencies'])
                            else:
                                print("Warning: data variable '" + 
                                str(dep) + "' that is given in the " + 
                                "dependency list is not found in the " +
                                "data dictionary. This dependency " +
                                "specification will be ignored.") 
                # Add the dependency specifications to the run_once
                # data that are not single numbers.
                for list_idx, name_list in enumerate(names):
                    for name_idx, name in enumerate(name_list):
                        data[name] = {'Value': values[list_idx][name_idx],
                                      'Type': 'Independent'}
                # Save the data.
                self._save_data(data)
            else:
                print('There is no data to save!')
       
        # Restore previous sweep variable values.
        for name in unique_names:
            self.value(name, prev_vals[name])
        
        self.exit_expt()
        
        self.rm_var('Runs')
        
        sec = time.time() - self._sweep_start_time
        hrs = int(sec / 3600.)
        min = int((sec - float(hrs) * 3600.) / 60.)
        sec = sec - float(hrs) * 3600. - float(min) * 60.
        print('The sweep execution time is %d:%02d:%06.3f.'
                %(hrs, min, sec))

    ###KEYBOARD LISTENERS###############################################
    def _listen_to_keyboard(self, 
            recog_keys=[27, 83, 115, 84, 116, 79, 111, 88, 120],
            clear_buffer=True):
        """
        Listen to the keyboard and determine if a specific key is
        pressed.
        
        Inputs:
            recog_kyes: list of keys to listen to.
            clear_buffer: if True analyze only the first key pressed and
                clear buffer afterwards.
        Output:
            None.
        """
        if kbhit():
            # Analyze the first character in the keyboard buffer.
            key = getch()
            if ord(key) in recog_keys:
                if ord(key) == 27:
                    # [ESC] is pressed.
                    self._sweep_status= 'abort'
                    self._sweep_msg = 'The experiment has been aborted!'
                elif ord(key) == 83 or ord(key) == 115:
                    # Either [S] or [s] is pressed.
                    self._sweep_status= 'abort-and-save'
                    self._sweep_msg = 'The experiment has been aborted!'
                elif ord(key) == 84 or ord(key) == 116:
                    # Either [T] or [t] is pressed.
                    if self._sweep_pts_acquired > 0:
                        finish_time = \
                                time.localtime(self._sweep_start_time + 
                                self._sweep_number_of_pts * 
                                (time.time() - self._sweep_start_time) /
                                self._sweep_pts_acquired)
                        self._sweep_msg = ('The estimated finish ' +
                                'time is ' + 
                                time.strftime("%H:%M:%S", finish_time) +
                                ' on ' + 
                                time.strftime("%m/%d/%Y", finish_time) +
                                '.')
                    else:
                        self._sweep_msg = ('Not enough data ' +
                        'acquired to allow the time estimation.')
                elif ord(key) == 79 or ord(key) == 111:
                    # Either [O] or [o] is pressed.
                     self._standard_output = not self._standard_output
                elif ord(key) == 88 or ord(key) == 120:
                    # Either [X] or [x] is pressed.
                     self._sweep_msg = ('Hey, stop wondering! ' +
                                        'Get back to work!')
        
        # Clear the keyboard buffer if requested.
        if clear_buffer:
            while kbhit():
                getch()

    ###PLOTTING METHODS#################################################
    def _in_brackets(self, v):
        """
        Return variable units in brackets.
        
        Input:
            v: value or name of the variable.
        Output:
            units: the variable units in brackets.
        """
        unit = self.get_units(v)
        if unit != '':
            return ' [' + unit + ']'
        else:
            return unit
    
    def _init_1d_plot(self, names, values, data, plot_data_vars):  
        # Make a list of the data variables that should be plotted.
        if plot_data_vars is not None:
            for var in self._comb_strs(plot_data_vars):
                if var not in data:
                    print("Warning: variable '" + var + 
                    "' is not found among the data dictionary keys: " + 
                    str(data.keys()) + ". This data will not be plotted.")
            plot_data_vars = [var for var in self._comb_strs(plot_data_vars) 
                    if var in data and 
                    np.ndim(data[var]['Value']) == 1]
            # Determine the X axis label.
            if plot_data_vars and values[0][0].size > 1:
                self._same_x_axis = True
                self._similar_x_axis = True
                for p_idx in range(len(names)):
                    if names[p_idx][0] != names[0][0]:
                        self._same_x_axis = False
                    if ((values[p_idx][0] != values[0][0]).any() or 
                        self.get_units(self.value(names[p_idx][0])) != 
                        self.get_units(self.value(names[0][0]))):
                        self._similar_x_axis = False
                    if (self._same_x_axis == False and 
                            self._similar_x_axis == False):
                        break
                if self._same_x_axis:
                    self._create_1d_plot(names[0][0], values[0][0], data,
                            plot_data_vars)
                elif self._similar_x_axis:
                    self._create_1d_plot([nm[0] for nm in names], 
                            values[0][0], data, plot_data_vars)
                else:
                    self._create_1d_plot('Run Iteration - Fast Axis', 
                            np.array(range(values[0][0].size)),
                            data, plot_data_vars)
            else:
                plot_data_vars = None  
            
        return plot_data_vars

    def _create_1d_plot(self, independent_vars, values, data, plot_data_vars):
        # Specify x-axis label.
        xlabel = ''
        for var in self._comb_strs(independent_vars):
            if var != 'Run Iteration - Fast Axis':
                xlabel = xlabel + var + self._in_brackets(var) + ', '
            else:
                xlabel = xlabel + var + ', '
        xlabel = xlabel[:-2]
        
        # Specify y-axis label.
        ylabel = ''
        same_y_units = False
        if len(set([self.get_units(data[var]['Value'])
                for var in plot_data_vars])) == 1:
            same_y_units = True
        else:
            same_y_units = False
        for var in plot_data_vars:
            if ('Preferences' in data[var] and
                    'name' in data[var]['Preferences']):
                name = data[var]['name']
            else:
                name = var
            if same_y_units:
                ylabel = (ylabel + name + ', ')
            else:
                ylabel = (ylabel + name + 
                        self._in_brackets(data[var]['Value']) + ', ')
        ylabel = ylabel[:-2]
        if same_y_units:
            ylabel = ylabel + \
                    self._in_brackets(data[plot_data_vars[0]]['Value'][0])
 
        # Initialize the plot.
        plt.figure(1)
        plt.ioff()
        plt.clf()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Specify line styles.
        self.plot_lines = {}
        for var in plot_data_vars:
            # Default line styles and plot parameters. Predefining helps
            # make the code below a bit shorter and can potentially
            # reduce severity of the bugs here.
            linestyle = 'b-'
            linewidth = 2
            linelabel = var
            # Redefine any plot parameters that are actually specified.
            if var in data and 'Preferences' in data[var]:
                if 'linestyle' in data[var]['Preferences']:
                    linestyle = data[var]['Preferences']['linestyle']
                if 'linewidth' in data[var]['Preferences']:
                    linewidth = data[var]['Preferences']['linewidth']
                if 'legendlabel' in data[var]['Preferences']:
                    linelabel = data[var]['Preferences']['legendlabel']
            self.plot_lines[var], = plt.plot(values, np.zeros_like(values), 
                linestyle, lw=linewidth, label=linelabel)
        if len(plot_data_vars) > 1:
            plt.legend() 
        
        # Specify axes limits.
        plt.xlim(min(values[0], values[1]), max(values[0], values[1]))  
        self.plot_ymin = None
        self.plot_ymax = None
        for var in plot_data_vars:
            if (var in data and 'Preferences' in data[var] and
               'ylim' in data[var]['Preferences']):
                if data[var]['Preferences']['ylim'][0] is not None:
                    if self.plot_ymin is not None:
                        self.plot_ymin = np.min([self.plot_ymin,
                            data[var]['Preferences']['ylim'][0]])
                    else:
                        self.plot_ymin = data[var]['Preferences']['ylim'][0]
                if data[var]['Preferences']['ylim'][1] is not None:
                    if self.plot_ymax is not None:    
                        self.plot_ymax = np.max([self.plot_ymax,
                            data[var]['Preferences']['ylim'][1]])
                    else:
                        self.plot_ymax = data[var]['Preferences']['ylim'][1]
        if self.plot_ymin is not None and self.plot_ymax is not None:
            plt.ylim(self.plot_ymin, self.plot_ymax)

    def _update_1d_plot(self, independent_vars, values, data, 
            plot_data_vars, idx):
        # We need at least two points.
        if idx == 0:
            return
        # Specify X axis values and names.
        if self._same_x_axis:
            values = values[0][0][:idx+1]
        elif self._similar_x_axis:
            values = values[0][0][:idx+1]
        else:
            values = np.array(range(idx + 1))

        plt.figure(1)

        # Set data.
        for var in plot_data_vars:
            self.plot_lines[var].set_ydata(data[var]['Value'])
        
        # Specify axes limits.
        plt.xlim(min(values), max(values))
        if self.plot_ymax is None:
            ymax = np.max([np.max(data[var]['Value'][0:len(values)])
                    for var in plot_data_vars])
        else:
            ymax = self.plot_ymax
        if self.plot_ymin is None:
            ymin = np.min([np.min(data[var]['Value'][0:len(values)])
                    for var in plot_data_vars])
        else:
            ymin = self.plot_ymin
        if ymin == ymax:
            ymax = ymax + np.finfo(type(ymax)).eps
        plt.ylim(ymin, ymax)
        
        # Redraw.
        plt.draw()
        plt.pause(0.05)