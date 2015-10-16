# Copyright (C) 2015 Ivan Pechenezhskiy
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

"""
This module contains simplified interface to some specific 
experiment resources. The classes defined here are intended to be used
with Experiment class from the experiment module.
"""

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
import itertools

from labrad.server import inlineCallbacks
import labrad.units as units

import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq


class ResourceDefinitionError(Exception): pass
class ResourceInitializationError(Exception): pass

class GHzFPGABoards(object):
    """GHz FPGA boards simplified interface."""
    def __init__(self, cxn, resource):
        """
        Initialize GHz FPGA boards.
        
        Input:
            cxn: LabRAD connection object.
            resource: resource dictionary.
            var: name of the variable.
        Output:
            None.
        """
        if 'Server' in resource:
            self.server_name = resource['Server']
        else:
            self.server_name = 'GHz FPGAs'
        self.server = cxn[self.server_name]
        
        self._init_boards(cxn, resource)
        
    @inlineCallbacks
    def _init_boards(self, cxn, resource):
        """Initialize GHz FPGA boards."""
        # Check the boards specification.
        if 'Boards' in resource:
            boards = resource['Boards']
            if isinstance(boards, str):
                self.boards = [boards]
            elif (isinstance(boards, list) and
                    all([isinstance(board, str) for board in boards])):
                self.boards = boards
            else:
                raise ResourceDefinitionError("GHz FPGA boards " +
                        "in the resource dictionary should be " +
                        "specified as a string or a list of strings.")
        else:
            raise ResourceDefinitionError("'Boards' field is not found " +
                    " in the experiment resource: " + str(resource) + ".")

        
        # Get the board settings from the resource specifications.
        self.consts = {}
        self.dacs = []
        self.adcs = []
        self.dac_settings = []
        self.adc_settings = []
        self._results = []
        
        if not boards:
            return
        
        self._data_dacs = []
        self._data_adcs = []
        for board in boards:
            if board not in resource:
                raise ResourceDefinitionError("Settings for board '" + 
                        board + "' are not present in the experiment " +
                        "resource: " + str(resource) + ".")
            settings = resource[board]
            if board.find('DAC') != -1:
                self.dacs.append(board)
                for ch in ['DAC A', 'DAC B']:
                    if ch not in settings or settings[ch] is None:
                        settings[ch] = 'None'
                    elif not isinstance(settings[ch], str):
                        raise ResourceDefinitionError("Board '" +
                                board + "' '" + ch + "' setting" +
                                " should be specified either as " + 
                                " a string or be of a None type.")
                self.dac_settings.append(settings)
                if 'Data' in settings and settings['Data']:
                    self._data_dacs.append(board)
            elif board.find('ADC') != -1:
                self.adcs.append(board)
                self.adc_settings.append(settings)
                if 'Data' in settings and settings['Data']:
                    self._data_adcs.append(board)
            else:
                raise ResourceDefinitionError("Neither 'DAC' nor 'ADC'" +
                        "string is found in board name '" +
                        board + "'.")
        if self._data_dacs and self._data_adcs:
            raise ResourceDefinitionError("Either DAC or ADC boards " +
                    "must return the data, not both.")
        
        # Check that all DAC and ADC boards are unique.
        if len(self.dacs) != len(set(self.dacs)):
            raise ExperimentDefinitionError("All DAC boards must have" +
            " unique names in the resource dictionary. The following" + 
            " DAC boards are given: " + str(dacs) + ".")
        if len(self.adcs) != len(set(self.adcs)):
            raise ExperimentDefinitionError("All ADC boards must have" +
            " unique names in the resource dictionary. The following" +
            " ADC boards are given: " + str(self.adcs) + ".")

        # Check that the boards are listed on the GHz FPGA server.
        p = self.server.packet()
        listed_boards = (yield p.list_devices().send())['list_devices']
        listed_boards = [board for idx, board in listed_boards]
        for board in boards:
            if board not in listed_boards:
                raise ResourceDefinitionError("Board '" + board +
                        "' is not found on server '" + 
                        self.server_name + "'.")

        # Get the board build constants from the LabRAD Registry.
        yield cxn.registry.cd(['', 'Servers', self.server_name])
        if self.dacs:
            consts = yield cxn.registry.get('dacBuild8')
            for name, value in consts:
                self.consts[name] = value
        if self.adcs:
            consts = yield cxn.registry.get('adcBuild1')
            for name, value in consts:
                self.consts[name] = value
        board_groups = yield cxn.registry.get('boardGroups')
        for board_group in board_groups:
            board, server, port, group = board_group
            for board_delay in group:
                name_addr, delay = board_delay
                name = board + ' ' + name_addr
                if name in self.dacs:
                    k = [idx for idx, dac in enumerate(self.dacs) 
                        if name == dac][0]
                    self.dac_settings[k]['CalibDelay'] = delay 
                if name in self.adcs:
                    k = [idx for idx, adc in enumerate(self.adcs) 
                        if name == adc][0]
                    self.adc_settings[k]['CalibDelay'] = delay
        if self._data_dacs:
            try:
                preamp_timeout = yield cxn.registry.get('PREAMP_TIMEOUT')
            except:
                print("'PREAMP_TIMEOUT' key is not found in the " +
                        "LabRAD Registry. It will be set to 1253 " +
                        "PreAmpTimeCounts.")
                preamp_timeout = 1253
            self.consts['PREAMP_TIMEOUT'] = (preamp_timeout *
                    units.PreAmpTimeCounts)
        try:
            dac_zero_pad_len = yield cxn.registry.get('DAC_ZERO_PAD_LEN')
        except:
            print("'DAC_ZERO_PAD_LEN' key is not found in the " +
                    "LabRAD Registry. It will be set to 10 ns.")
            dac_zero_pad_len = 10
        self.consts['DAC_ZERO_PAD_LEN'] = dac_zero_pad_len * units.ns
        
        # Create a list that contains all requested/used waveforms.
        self.requested_waveforms = [settings[ch] for settings in
                self.dac_settings for ch in ['DAC A', 'DAC B']]
                   

    def process_waveforms(self, waveforms):
        """
        Check whether the specified waveforms with the waveforms defined
        in the run_once method. Get SRAMs from the waveforms.
        
        Input:
            waveforms: dictionary with the waveforms.
        Output:
            dac_srams: list of DAC SRAMs.
            sram_length: SRAM length.
            sram_delay: SRAM delay.
        """
        for idx, settings in enumerate(self.dac_settings):
            for channel in ['DAC A', 'DAC B']:
                if self.dac_settings[idx][channel] not in waveforms:
                    raise ResourceDefinitionError("'" + 
                        str(self.dacs[idx]) +
                        "' '" + str(channel) + "' waveform setting: '" +
                        self.dac_settings[idx][channel] +
                        "' is not recognized.")
        
        dac_srams = [seq.waves2sram(waveforms[self.dac_settings[k]['DAC A']], 
                                    waveforms[self.dac_settings[k]['DAC B']])
                                    for k, dac in enumerate(self.dacs)]
        sram_length = len(waveforms[self.dac_settings[0]['DAC A']])
        sram_delay = np.ceil(sram_length / 1000)
        
        return dac_srams, sram_length, sram_delay
    
    def get_adc(self, adc=None):
        """
        If only a single ADC board is present, return its name. If more
        than one board is present, check that a board with a a given name
        actually exists, otherwise raise an error. Return the board index
        as a second parameter.
        
        Input:
            adc (optional): ADC board name (default: None).
        Output:
            adc: ADC board name.
        """
        if len(self.adcs) == 1:
            return self.adcs[0], 0
        elif adc is None:
            raise Exception("The ADC board name should be explicitly " +
                "specified since more than one ADC board is present.")
        elif adc not in self.adcs:
            raise Exception("ADC board '" + str(adc) + "' is not found.")
        return adc, self.adcs.index(adc)
 
    def set_adc_setting(self, setting, value, adc=None):
        """
        Change one of the ADC settings.
        
        Inputs:
            setting: name of setting you want to change.
            value: value to change the setting to.
            adc: ADC board name. If None and only one board in is
            present the board name will be automatically recognized.
        Output:
            None.
        """
        adc, idx = self.get_adc(adc)
        
        if setting in self.adc_settings[idx]:
            self.adc_settings[idx][setting] = value
        else:
            raise Exception("'" + str(setting) + 
                    "' is not a valid ADC setting.")

    def get_adc_setting(self, setting, adc=None):
        """
        Get an ADC setting.
        
        Inputs:
            setting: name of setting you want to change.
            adc: ADC board name. If None and only one board in is
            present the board name will be automatically recognized.
        Output:
            value: value of the ADC setting.
        """
        adc, idx = self.get_adc(adc)
        
        if setting in self.adc_settings[idx]:
            return self.adc_settings[idx][setting]
        else:
            raise Exception("'" + str(setting) + 
                    "' is not a valid ADC setting.")

    def load_dacs(self, sram, memory):
        """Load DACs with Memory commands and SRAM."""
        for k, dac in enumerate(self.dacs):
            p = self.server.packet()
            p.select_device(dac)
            p.memory(memory[k])
            # p.start_delay(self.dac_settings[k]['CalibDelay'])
            # Handle dual block calls here, in a different way than Sank
            # did. This should be compatible.
            if len(sram[k]) > self.consts['SRAM_LEN']:
                # Shove last chunk of SRAM into BLOCK1, be sure this can 
                # contain what you need it to contain.
                sram1 = sram[k][-self.consts['SRAM_BLOCK1_LEN']:]
                # Amount of SRAM that's extra.
                sram_diff = len(sram[k]) - self.consts['SRAM_LEN']
                # Calculate the number of the delay blocks:
                # sram_diff = x * 'SRAM_DELAY_LEN' + y.
                x, y = divmod(sram_diff, self.consts['SRAM_DELAY_LEN'])
                if y == 0:
                    delay_blocks = x
                else:
                    delay_blocks = x + 1 # Overshoot. 
                sram0 = sram[k][:(self.consts['SRAM_BLOCK0_LEN'] + sram_diff -
                        delay_blocks * self.consts['SRAM_DELAY_LEN'])]
                if len(set(sram[k][(self.consts['SRAM_BLOCK0_LEN'] + sram_diff -
                        delay_blocks * self.consts['SRAM_DELAY_LEN']) - 
                        4:len(sram[k]) - self.consts['SRAM_BLOCK1_LEN']])) != 1:
                    # Ensure that the delay block is constant.
                    raise Exception('Dual block mode will not work for ' +
                            'the requested pulse sequence.')
                p.sram_dual_block(sram0, sram1, delay_blocks * 
                        self.consts['SRAM_DELAY_LEN'])
            else:
                p.sram(sram[k])
            self._results.append(p.send(wait=False))
            
    def load_adcs(self):
        """Load ADCs with correct variables."""
        for idx, adc in enumerate(self.adcs):
            p = self.server.packet()
            p.select_device(adc)
            p.start_delay(int((self.adc_settings[idx]['ADCDelay']['ns']) / 4) + 
                    self.adc_settings[idx]['CalibDelay'])
            p.adc_run_mode(self.adc_settings[idx]['RunMode'])
            p.adc_filter_func(self.filter_bytes(self.adc_settings[idx]), 
                    int(self.adc_settings[idx]['FilterStretchLen']['ns']),
                    int(self.adc_settings[idx]['FilterStretchAt']['ns']))
            dPhi = int(self.adc_settings[idx]['DemodFreq']['Hz'] / 7629)
            phi0 = int(self.adc_settings[idx]['DemodPhase']['rad'] * (2**16))
            for k in range(self.consts['DEMOD_CHANNELS']):
                p.adc_demod_phase(k, dPhi, phi0)
                p.adc_trig_magnitude(k, self.adc_settings[idx]['DemodSinAmp'],
                        self.adc_settings[idx]['DemodCosAmp'])
            self._results.append(p.send(wait=False))
        
    def filter_bytes(self, settings):
        """Set the filter for a specific experiment."""
        # ADC collects at a 2 ns acquisition rate, but the filter function
        # has a 4 ns resolution.
        filter_func = settings['FilterType'].lower()
        sigma = settings['FilterWidth']['ns']
        window = np.zeros(int(settings['FilterLength']['ns'] / 4))
        if filter_func == 'square':
            window = window + (128)
            filt = np.append(window, np.zeros(self.consts['FILTER_LEN'] -
                    len(window)))
            filt = np.array(filt, dtype='<u1')
        elif filter_func == 'gaussian':
            env = np.linspace(-0.5, 0.5, len(window))
            env = np.floor(128 * np.exp(-((env / (2 * sigma))**2)))
            filt =  np.append(env, np.zeros(self.consts['FILTER_LEN'] -
                    len(env)))
            filt = np.array(filt, dtype='<u1')        
        elif filter_func == 'hann':
            env = np.linspace(0, len(window) - 1, len(window))
            env = np.floor(128 * np.sin(np.pi * env / (len(window) - 1))**2)
            filt =  np.append(env, np.zeros(self.const['FILTER_LEN'] - len(env)))
            filt = np.array(filt, dtype='<u1')
        elif filter_func == 'exp':
            env = np.linspace(0,(len(window) - 1) * 4, len(window))
            env = np.floor(128 * np.exp(-env / sigma))
            filt =  np.append(env, np.zeros(self.consts['FILTER_LEN'] -
                    len(env)))
            filt = np.array(filt, dtype='<u1')
        else:
            raise Exception('Filter function %s not recognized.'
                    %filter_func)
        return filt.tostring()

    def load(self, dac_srams, dac_mems):
        """
        Load FPGA boards with the required memory and settings. Input 
        arguments, dac_srams and dac_mems should be lists 
        which correspond to the SRAM and Memory in the order in which
        the DACs are defined in the experiment resource dictionary.
        The first listed DAC is always assumed to be the master. 
        
        Inputs:
            sram: list of DAC SRAM waves. Use 
                ghz_self.boards_control.waves2sram method to get the right 
                format.
            memory: list of memory commands. Use memory tools in
                DAC_control to build a memory sequence.
        Output:
            run_data: returns the result of the self.boards.run_sequence 
                command.
        """
        if len(dac_mems) != len(dac_srams):
            raise Exception('Not enough memory commands to ' +
                    'populate the boards!')
        
        self.load_dacs(dac_srams, dac_mems)
        
        if self._data_adcs:
            p = self.server.packet()
            # Determine which set of boards to run, not the order.
            p.daisy_chain(list(itertools.chain(*[self.dacs, self.adcs])))
            timing_order_list = []
            for adc in self._data_adcs:
                if (self.get_adc_setting('RunMode', adc).lower() ==
                        'average'):
                    timing_order_list.append(adc)
                elif (self.get_adc_setting('RunMode', adc).lower() ==
                        'demodulate'):
                    # Record channel 0.
                    timing_order_list.append(adc + '::0')
                else:
                    raise ResourceDefinitionError("ADC board '" +
                            adc + "' 'RunMode' setting " +
                            "should be either 'average'" +
                            " or 'demodulate'.")
            p.timing_order(timing_order_list)
            self._results.append(p.send(wait=False))
            self.load_adcs()
        elif self._data_dacs:
            p = self.server.packet()
            p.daisy_chain(list(itertools.chain(*[self.dacs])))
            p.timing_order(self._data_dacs)
            self._results.append(p.send(wait=False))
            
        # Ensure that all packets are sent.
        for result in self._results:
            result.wait()
        self._results = []
    
    def run(self, reps=1020):
        """
        Execute the run sequence a set number of times.
        
        Input:
            reps: number of repetitions in the sequence (default: 1020).
        Output:
            run_data: returns the result of the self.boards.run_sequence 
                command.
        """
        return self.server.run_sequence(int(reps), (bool(self._data_dacs) or 
                                                    bool(self._data_adcs)))
                                               
    def load_and_run(self, dac_srams, dac_mems, reps=1020):
        """
        Load FPGA boards with the required memory and settings, and 
        execute the run sequence a set number of times. This method
        should be called at the end of each run_once. Input arguments,
        dac_srams and dac_mems should be lists which correspond to the 
        SRAM and Memory in the order in which the DACs are defined in
        the experiment resource dictionary. The first listed DAC is
        always assumed to be the master. 
        
        Inputs:
            sram: list of DAC SRAM waves. Use 
                ghz_self.boards_control.waves2sram method to get the right 
                format.
            memory: list of memory commands. Use memory tools in
                DAC_control to build a memory sequence.
            reps: number of repetitions in the sequence (default: 1020).
        Output:
            run_data: returns the result of the self.boards.run_sequence 
                command.
        """
        self.load(dac_srams, dac_mems)
        return self.run(reps)
        
    def init_mem_lists(self):
        """
        Initialize memory command lists. The output is a list with the
        length that corresponds to the number of DAC boards. Each list
        item is a list on its own that should be populated with the
        memory commands corresponding to the board. The memory command
        format is described in
        Servers.Instruments.GHzBoards.command_sequences.
        
        Input:
            None.
        Output:
            mem_lists: list of memory command lists.
        """
        mem_lists = [[] for dac in self.dacs]
        for idx, settings in enumerate(self.dac_settings):
            if 'FO1 FastBias Firmware Version' in settings:
                mem_lists[idx].append({'Type': 'Firmware', 'Channel': 1, 
                              'Version': settings['FO1 FastBias Firmware Version']})
            if 'FO2 FastBias Firmware Version' in settings:
                mem_lists[idx].append({'Type': 'Firmware', 'Channel': 2, 
                              'Version': settings['FO2 FastBias Firmware Version']})
        return mem_lists


class BasicInterface(object):
    """
    Basic interface class.
    """
    def __init__(self, cxn, res, var):
        """
        Initialize a resource.
        
        Input:
            cxn: LabRAD connection object.
            res: resource dictionary.
            var: name of the variable.
        Output:
            None.
        """
        self._initialized = False
        self._request_sent = False
        self._res = res
        self._var = var
        self._setting = None
        self.server = self._get_server(cxn)
        
        try:
            self._init_resource()
        except:
            raise ResourceInitializationError('Resource ' +
                    str(resource) + ' could not be intialized.')
        
        self._initialized = True
    
    def __exit__(self, type, value, traceback):
        """Properly exit the resource."""
        pass

    def _init_resource(self):
        """Do device specific initialization."""
        pass
    
    def _get_server(self, cxn):       
        """Get server connection object."""
        if 'Server' in self._res:
            try:
                return cxn[self._res['Server']]
            except:
                raise ResourceDefinitionError("Could not connect to " +
                    "server '" + str(self._res['Server']) + "'.")
        else:
            raise ResourceDefinitionError("Key 'Server' is not found" +
                    " in resource: " + str(self._res) + ".")
    
    def send_request(self, value=None):
        """Send an empty non-blocking request."""
        if self._initialized:
            p = self.server.packet()
            if self._setting is not None:
                if value is None:
                    p[self._setting]()
                else:
                    p[self._setting](value)
            self._result = p.send(wait=False)
            self._request_sent = True
        else:
            raise ResourceInitializationError("Resource " +
                    str(self._res) + " is not properly initialized.")
        
    def acknowledge_request(self):
        """Wait for the result of a non-blocking request."""
        if self._initialized and self._request_sent:
            self._request_sent = False
            if self._setting is not None:
                return self._result.wait()[self._setting]
            else:
                return self._result.wait()

class GPIBInterface(BasicInterface):
    """
    Simplified GPIB interface class.
    """
    @inlineCallbacks
    def _init_resource(self):
        """Initialize a GPIB resource."""
        p = self.server.packet()
        devices = (yield p.list_devices().send())['list_devices']
        devices = [dev for id, dev in devices]
        if 'Address' in self._res:
            if self._res['Address'] in devices:
                self.address = self._res['Address']
            else:
                raise ResourceDefinitionError("Device with address '" +
                    str(self._res['Address']) + "' is not found.")
        elif len(devices) == 1:
            self.address = devices[0][0]
        else:
            raise ResourceDefinitionError("'Address' field is absent " +
                    " in the experiment resource: " +
                    str(self._res) + ".")
        
        if len(devices) == 1:
            self._single_device = True
        else:
            self._single_device = False
            
        self._init_gpib_resource()
    
    def _init_gpib_resource(self):
        """Do variable specific initialization."""
        pass
        
    def send_request(self, value=None):
        """Send a request to set a setting."""
        if self._initialized:
            p = self.server.packet()
            if not self._single_device:
                p.select_device(self.address)
            if self._setting is not None:
                p[self._setting](value)
            self._result = p.send(wait=False)
            self._request_sent = True
        else:
            raise ResourceDefinitionError("Resource '" +
                    str(self._res) + "' is not properly initialized.")
            

class RFGenerator(GPIBInterface):
    """
    GPIB RF generator simplified interface.
    """
    @inlineCallbacks
    def __exit__(self, type, value, traceback):
        """Turn the RF generator off and deselect it."""
        if hasattr(self, 'address'):
            p = self.server.packet()
            yield p.select_device(self.address).output(False).deselect_device().send()
    
    def _get_server(self, cxn):       
        """Get server connection object."""
        if 'Server' in self._res:
            server_name = self._res['Server']
        else:
            server_name = 'GPIB RF Generators'
        try:
            return cxn[server_name]
        except:
            raise ResourceDefinitionError("Could not connect to " +
                "server '" + server_name + "'.")

    @inlineCallbacks
    def _init_gpib_resource(self):
        """Initialize an RF generator."""
        if ('Variables' in self._res and 
                self._var in self._res['Variables'] and 
                isinstance(self._res['Variables'], dict) and 
                'Setting' in self._res['Variables'][self._var]):
            self._setting = self._res['Variables'][self._var]['Setting']
        elif self._var.lower().find('freq') != -1:
            self._setting = 'Frequency'
        elif self._var.lower().find('power') != -1:
            self._setting = 'Power'
        else:
            raise ResourceDefinitionError("Setting responsible for " +
                    "variable '" + self._var + "' is not specified " +
                    "in the experiment resource: " + 
                    str(self._res) + ".")
        
        p = self.server.packet()
        yield p.select_device(self.address).reset().send()
        self._output_set = False

    def send_request(self, value):
        """Send a request to set a setting."""
        if self._initialized:
            if value is not None:
                p = self.server.packet()
                if not self._single_device:
                    p.select_device(self.address)
                p[self._setting](value)
                if not self._output_set:
                    p.output(True)
                    self._output_set = True
                self._result = p.send(wait=False)
                self._request_sent = True
        else:
            raise ResourceDefinitionError("Resource " +
                    str(self._res) + " is not properly initialized.")


class SIM928VoltageSource(GPIBInterface):
    """
    SRS SIM928 voltage source simplified interface.
    """    
    @inlineCallbacks
    def __exit__(self, type, value, traceback):
        """Turn the voltage source off and deselect it."""
        if hasattr(self, 'address'):
            p = self.server.packet()
            yield p.select_device(self.address).voltage(0 * units.V).deselect_device().send()

    def _get_server(self, cxn):       
        """Get server connection object."""
        if 'Server' in self._res:
            server_name = self._res['Server']
        else:
            server_name = 'SIM928'
        try:
            return cxn[server_name]
        except:
            raise ResourceDefinitionError("Could not connect to " +
                "server '" + server_name + "'.")

    @inlineCallbacks
    def _init_gpib_resource(self):
        """Initialize a voltage source."""
        if self._single_device:
            p = self.server.packet()
            yield p.select_device(self.address).send()
        self._output_set = False
        
    def send_request(self, voltage):
        """Send a request to set the output voltage."""
        if self._initialized:
            if voltage is not None:
                p = self.server.packet()
                if not self._single_device:
                    p.select_device(self.address)
                p.voltage(voltage)
                if not self._output_set:
                    p.output(True)
                    self._output_set = True
                self._result = p.send(wait=False)
                self._request_sent = True
        else:
            raise ResourceDefinitionError("Resource " +
                    str(self._res) + " is not properly initialized.")


class LabBrickAttenuator(BasicInterface):
    """
    Lab Brick attenuator simplified interface.
    """
    @inlineCallbacks
    def __exit__(self, type, value, traceback):
        """Deselect the attenuator."""
        if self._initialized:
            p = self.server.packet()
            yield p.deselect_attenuator().send()
    
    def _get_server(self, cxn):       
        """Get server connection object."""
        if 'Server' in self._res:
            server_name = self._res['Server']
        else:
            server_name = (os.environ['COMPUTERNAME'].lower() +
                                ' Lab Brick Attenuators')
        try:
            return cxn[server_name]
        except:
            raise ResourceDefinitionError("Could not connect to " +
                "server '" + server_name + "'.")
    
    @inlineCallbacks
    def _init_resource(self):
        """Initialize a Lab Brick attenuator."""
        p = self.server.packet()
        devices = (yield p.list_devices().send())['list_devices']
        if 'Serial Number' in self._res:
            if self._res['Serial Number'] in devices:
                self.address = self._res['Serial Number']
            else:
                raise ResourceDefinitionError("Device with serial " +
                    "number " + str(self._res['Serial Number']) +
                    " is not found.")
        elif len(devices) == 1:
            self.address = devices[0][0]
        else:
            raise ResourceDefinitionError("'Serial Number' field is " +
                    "absent in the experiment resource: " +
                    str(self._res) + ".")
        
        if len(devices) == 1:
            self._single_device = True
            p = self.server.packet()
            yield p.select_attenuator(self.address).send()
        else:
            self._single_device = False
        
    def send_request(self, attenuation):
        """Send a request to set the attenuation."""
        if self._initialized:
            if attenuation is not None:
                p = self.server.packet()
                if not self._single_device:
                    p.select_attenuator(self.address)
                self._result = p.attenuation(attenuation).send(wait=False)
                self._request_sent = True
        else:
            raise ResourceDefinitionError("Resource " +
                    str(self._res) + " is not properly initialized.")


class ADR3(BasicInterface):
    """
    ADR3 simplified interface for temperature monitoring.
    """
    def _get_server(self, cxn):       
        """Get server connection object."""
        if 'Server' in self._res:
            server_name = self._res['Server']
        else:
            server_name = 'ADR3'
        try:
            return cxn[server_name]
        except:
            raise ResourceDefinitionError("Could not connect to " +
                "server '" + server_name + "'.")
    
    def _init_resource(self):
        """Initialize the temperature variable."""
        if ('Variables' in self._res and 
                self._var in self._res['Variables'] and 
                isinstance(self._res['Variables'], dict)):
            var_dict = True
        else:
            var_dict = False
        
        if var_dict and 'Setting' in self._res['Variables'][self._var]:
            self._setting = self._res['Variables'][self._var]['Setting']
        else:
            self._setting = 'Temperatures'
        if var_dict and 'Stage' in self._res['Variables'][self._var]:
            if self._res['Variables'][self._var]['Stage'].lower().find('50k') != -1:
                self._temp_idx = 0
            elif self._res['Variables'][self._var]['Stage'].lower().find('3k') != -1:
                self._temp_idx = 1
            elif self._res['Variables'][self._var]['Stage'].lower().find('ggg') != -1:
                self._temp_idx = 2
            elif self._res['Variables'][self._var]['Stage'].lower().find('faa') != -1:
                self._temp_idx = 3
            else:
                self._temp_idx = 3

        
    def acknowledge_request(self):
        """Wait for the result of a non-blocking request."""
        if self._request_sent:
            self._request_sent = False
            temperatures = self._result.wait()[self._setting]
            return temperatures[self._temp_idx]


class Leiden(BasicInterface):
    """
    Leiden simplified interface for temperature monitoring.
    """        
    def _get_server(self, cxn):       
        """Get server connection object."""
        if 'Server' in self._res:
            server_name = self._res['Server']
        else:
            server_name = 'Leiden DR Temperature'
        try:
            return cxn[server_name]
        except:
            raise ResourceDefinitionError("Could not connect to " +
                "server '" + server_name + "'.")

    def _init_resource(self):
        """Initialize the temperature variable."""
        if ('Variables' in self._res and 
                self._var in self._res['Variables'] and 
                isinstance(self._res['Variables'], dict)):
            var_dict = True
        else:
            var_dict = False
        
        self._setting = None
        if var_dict and 'Setting' in self._res['Variables'][self._var]:
            self._setting = self._res['Variables'][self._var]['Setting']
        elif 'Chamber' in self._res:
            if self._res['Chamber'].lower().find('exch') != -1:
                self._setting = 'Exhange Temperature'
            elif self._res['Chamber'].lower().find('still') != -1:
                self._setting = 'Still Temperature'
        if self._setting is None:
            self._setting = 'Mix Temperature'