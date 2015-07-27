# Copyright (C) Ivan Pechenezhskiy
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

import itertools

from labrad.server import inlineCallbacks
import labrad.units as units

import LabRAD.Servers.Instruments.GHzBoards.command_sequences as seq

class ResourceDefinitionError(Exception): pass


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
                                " should either be specified either as " + 
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
        
        p = self.server.packet()
        listed_boards = (yield p.list_devices().send())['list_devices']
        listed_boards = [board for idx, board in listed_boards]
        for board in boards:
            if board not in listed_boards:
                raise ResourceDefinitionError("Board '" + board +
                        "' is not found on server '" + 
                        self.server_name + "'.")

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
                        "LabRAD Resitry. It will be set to 1253 " +
                        "PreAmpTimeCounts.")
                preamp_timeout = 1253
            self.consts['PREAMP_TIMEOUT'] = (preamp_timeout *
                    units.PreAmpTimeCounts)

    def load_dacs(self, memory, sram):
        """Load DACs with Memory commands and SRAM."""
        for k, dac in enumerate(self.dacs):
            p = self.server.packet()
            p.select_device(dac)
            p.memory(memory[k])
            # p.start_delay(DACDelays[k])
            # Handle dual block calls here, in a different way than Sank in 
            # the fpga_Server. This should be compatible.
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
        for k, adc in enumerate(self.adcs):
            p = self.server.packet()
            p.select_device(adc)
            p.start_delay(int((self.adc_settings[k]['ADCDelay']['ns']) / 4) + 
                    self.adc_settings[k]['CalibDelay'])
            p.adc_run_mode(self.adc_settings[k]['RunMode'])
            p.adc_filter_func(self.filter_bytes(self.adc_settings[k]), 
                    self.adc_settings[k]['FilterStretchLen']['ns'],
                    self.adc_settings[k]['FilterStretchAt']['ns'])
            for k in range(self.consts['DEMOD_CHANNELS']):
                dPhi = int(self.adc_settings[k]['DemodFreq']['MHz'] / 7629)
                phi0 = int(self.adc_settings[k]['DemodPhase']['rad'] * (2**16))
                p.adc_demod_phase(k, dPhi, phi0)
                p.adc_trig_magnitude(k, self.adc_settings[k]['DemodSinAmp'],
                        self.adc_settings[k]['DemodCosAmp'])
            self._results.append(p.send(wait=False))
        
    def filter_bytes(self, setting):
        """Set the filter for a specific experiment."""
        # ADC collects at a 2 ns acquisition rate, but the filter function
        # has a 4 ns resolution.
        filter_func = setting['FilterType'].lower()
        sigma = setting['FilterWidth']['ns']
        window = np.zeros(int(setting['FilterLength']['ns'] / 4))
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

    
    def load_and_run(self, waveforms, memory_dict_list, reps=1020):
        """
        Load FPGA boards with the required memory and settings, and 
        execute the run sequence a set number of times. This method
        should be called at the end of each run_once. Input arguments
        waveroms and memory should be lists which correspond to the the SRAM
        and Memory in the order in which they are defined in experiment
        resource dictionary. The first listed DAC is always assumed to
        be the master. 
        
        Inputs:
            sram: list of DAC SRAM waves. Use 
                ghz_fpga_control.waves2sram method to get the right 
                format.
            memory: list of memory commands. Use memory tools in
                DAC_control to build a memory sequence.
            reps: number of repetitions in the sequence (default: 1020).
        Output:
            run_data: returns the result of the fpga.run_sequence 
                command.
        """
        if len(memory_dict_list) != len(self.dacs):
            raise Exception('Not enough memory commands to ' +
                    'populate the boards!')

        dac_srams = []
        dac_mems = []
        for k, dac in enumerate(self.dacs):
            dac_srams.append(seq.waves2sram(waveforms[self.dac_settings[k]['DAC A']], 
                                            waveforms[self.dac_settings[k]['DAC B']]))
            dac_mems.append(seq.mem_from_list(memory_dict_list[k]))
        
        self.load_dacs(dac_mems, dac_srams)
        
        if self._data_adcs:
            p = self.server.packet()
            # Determine which set of boards to run, not the order.
            p.daisy_chain(list(itertools.chain(*[self.dacs, self.adcs])))
            timing_order_list = []
            for idx, adc in enumerate(self._data_adcs):
                if 'RunMode' in self.adc_setting[idx]:
                    if (self.adc_settings[idx]['RunMode'].lower() ==
                            'average'):
                        timing_order_list.append(adc)
                    elif (self.adc_settings[idx]['RunMode'].lower() ==
                            'demodulate'):
                        # Record channel 0.
                        timing_order_list.append(adc + '::0')
                    else:
                        raise ResourceDefinitionError("ADC board '" +
                                adc + "' 'RunMode' " +
                                "should be either 'average'" +
                                " or 'demodulate'.")
                else:
                    raise ResourceDefinitionError("'RunMode' field " +
                            " is not found in ADC board '" + adc + "'.")           
            p.timing_order(timing_order_list)
            self._results.append(p.send(wait=False))
            self.load_adcs()
        elif self._data_dacs:
            p = self.server.packet()
            p.daisy_chain(list(itertools.chain(*[self._data_dacs])))
            p.timing_order(self._data_dacs)
            self._results.append(p.send(wait=False))
        
        for result in self._results:
            result.wait()
        self._results = []

        return self.server.run_sequence(reps, (bool(self._data_dacs) or 
                                               bool(self._data_adcs)))

class RFGenerator(object):
    """
    GPIB RF generator simplified interface.
    """
    def __init__(self, cxn, resource, var):
        """
        Initialize an RF generator.
        
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
            self.server_name = 'GPIB RF Generators'
        self.server = cxn[self.server_name]
        
        self._init_device(resource)
    
    @inlineCallbacks
    def __exit__(self, type, value, traceback):
        """Deselect the RG generator and turn it off."""
        p = self.server.packet()
        yield p.deselect_device().output(False).send()
    
    @inlineCallbacks
    def _init_device(self, resource):
        """Initialize an RF generator."""
        p = self.server.packet()
        devices = (yield p.list_devices().send())['list_devices']
        if 'Address' in resource:
            if resource['Address'] in devices:
                self.address = resource['Address']
            else:
                raise ResourceDefinitionError("Device with address '" +
                    str(resource['Address']) + "' is not found on server '" +
                    self.server_name + "'.")
        elif len(devices) == 1:
            self.address = devices[0][0]
        else:
            raise ResourceDefinitionError("'Address' field is absent " +
                    " in the experiment resource: " + str(resource) + ".")
        
        if ('Variables' in resource and var in resource and 
                isinstance(resource[var], dict)):
            self._setting = resource['Variables'][var]
        elif var.lower().find('freq') != -1:
            self._setting = 'Frequency'
        elif var.lower().find('power') != -1:
            self._setting = 'Power'
        else:
            raise ResourceDefinitionError("Setting responsible for " +
                    "variable '" + var + "' is not specified in the " +
                    "experiment resource: " + str(resource) + ".")
        
        if len(devices) == 1:
            self._single_device = True
        else:
            self._single_device = False
        
        self._output_set = False
        
        p = self.server.packet()
        yield p.select_device(self.address).reset().send()

    def send_request(self, value):
        """Send a request to set a setting."""
        p = self.server.packet()
        if not self._single_device:
            p.select_device(self.address)
        if not self._output_set:
            p.output(True)
            self._output_set = True
        self._result = p[self._setting](value).send(wait=False)
        self._sent = True
        
    def acknowledge_request(self):
        """Wait for the result of a non-blocking request."""
        if self._sent:
            self._sent = False
            return self._result.wait()


class LabBrickAttenuator(object):
    """
    Lab Brick attenuator simplified interface.
    """
    def __init__(self, cxn, resource, var):
        """
        Initialize a Lab Brick attenuator.
        
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
            self.server_name = (os.environ['COMPUTERNAME'].lower() +
                                ' Lab Brick Attenuators')
        self.server = cxn[self.server_name]
        
        self._init_device(resource)
    
    @inlineCallbacks
    def __exit__(self, type, value, traceback):
        """Deselect the attenuator."""
        p = self.server.packet()
        yield p.deselect_device().send()
    
    @inlineCallbacks
    def _init_device(self, resource):
        """Initialize a Lab Brick attenuator."""
        p = self.server.packet()
        devices = (yield p.list_devices().send())['list_devices']
        if 'Address' in resource:
            if resource['Address'] in devices:
                self.address = resource['Address']
            else:
                raise ResourceDefinitionError("Device with address '" +
                    str(resource['Address']) + "' is not found on server '" +
                    self.server_name + "'.")
        elif len(devices) == 1:
            self.address = devices[0][0]
        else:
            raise ResourceDefinitionError("'Address' field is absent " +
                    " in the experiment resource: " + str(resource) + ".")
        
        if len(devices) == 1:
            self._single_device = True
        else:
            self._single_device = False
        
    def send_request(self, attenuation):
        """Send a request to set the attenuation."""
        p = self.server.packet()
        if not self._single_device:
            p.select_attenuator(self.address)
        self._result = p.attenuation(attenuation).send(wait=False)
        self._sent = True
        
    def acknowledge_request(self):
        """Wait for the result of a non-blocking request."""
        if self._sent:
            self._sent = False
            return self._result.wait()
            

class VoltageSource(object):
    """
    SRS SIM928 voltage source simplified interface.
    """
    def __init__(self, cxn, resource, var):
        """
        Initialize a voltage source.
        
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
            self.server_name = 'SIM928'
        self.server = cxn[self.server_name]
        
        self._init_device(resource)
    
    @inlineCallbacks
    def __exit__(self, type, value, traceback):
        """Deselect the voltage source and turn it off."""
        p = self.server.packet()
        yield p.deselect_device().output(False).send()
    
    @inlineCallbacks
    def _init_device(self, resource):
        """Initialize a voltage source."""
        p = self.server.packet()
        devices = (yield p.list_devices().send())['list_devices']
        if 'Address' in resource:
            if resource['Address'] in devices:
                self.address = resource['Address']
            else:
                raise ResourceDefinitionError("Device with address '" +
                    str(resource['Address']) + "' is not found on server '" +
                    self.server_name + "'.")
        elif len(devices) == 1:
            self.address = devices[0][0]
        else:
            raise ResourceDefinitionError("'Address' field is absent " +
                    " in the experiment resource: " + str(resource) + ".")
        
        if len(devices) == 1:
            self._single_device = True
        else:
            self._single_device = False
        
        self._output_set = False
        
        p = self.server.packet()
        yield p.select_device(self.address).reset().send()
        
    def send_request(self, voltage):
        """Send a request to set the output voltage."""
        p = self.server.packet()
        if not self._single_device:
            p.select_device(self.address)
        if not self._output_set:
            p.output(True)
            self._output_set = True
        self._result = p.voltage(voltage).send(wait=False)
        self._sent = True
        
    def acknowledge_request(self):
        """Wait for the result of a non-blocking request."""
        if self._sent:
            self._sent = False
            return self._result.wait()