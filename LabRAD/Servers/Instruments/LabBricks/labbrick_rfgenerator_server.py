# Copyright (C) 2015 Guilhem Ribeill
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
### BEGIN NODE INFO
[info]
name = Lab Brick RF Generators
version = 0.1
description =  Gives access to Lab Brick RF generators. This server self-refreshes.
instancename = %LABRADNODE% LBRF

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

import ctypes

from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.reactor import callLater
from twisted.internet.task import LoopingCall

from labrad.server import LabradServer, setting
from labrad.errors import DeviceNotSelectedError
import labrad.units as units

MAX_NUM_ATTEN = 64      # maximum number of connected RF Generators
MAX_MODEL_NAME = 32     # maximum length of Lab Brick model name

class LBRFGenServer(LabradServer):
    name='%LABRADNODE% LBRF'
    refreshInterval = 60
    defaultTimeout = 0.1 * units.s
    
    @inlineCallbacks
    def getRegistryKeys(self):
        '''Get registry keys for the Lab Brick RF Generator Server.'''
        reg = self.client.registry()
        yield reg.cd(['', 'Servers', 'Lab Brick RF Generators'], True)
        dirs, keys = yield reg.dir()
        if 'Lab Brick RF Generator DLL Path' not in keys:
            self.DLL_path = r'Z:\mcdermott-group\LabRAD\LabBricks\vnx_fmsynth.dll'
        else:
            self.DLL_path = yield reg.get('Lab Brick RF Generator DLL Path')
        print("Lab Brick RF Generator DLL Path is set to " + str(self.DLL_path))
        if 'Lab Brick RF Generator Timeout' not in keys:
            self.waitTime = self.defaultTimeout
        else:
            self.waitTime = yield reg.get('Lab Brick RF Generator Timeout')
        print("Lab Brick RF Generator Timeout is set to " + str(self.waitTime))
        if 'Lab Brick RF Generator Server Autorefresh' not in keys:
            self.autoRefresh = True
        else:
            self.autoRefresh = yield reg.get('Lab Brick RF Generator Server Autorefresh')
        print("Lab Brick RF Generator Server Autorefresh is set to " + str(self.autoRefresh))

    @inlineCallbacks    
    def initServer(self):
        '''Initialize the Lab Brick RF Generator Server.'''
        yield self.getRegistryKeys()
        try:
            self.VNXdll = yield ctypes.CDLL(self.DLL_path)
        except Exception:
            raise Exception('Could not find Lab Brick RF Generator DLL')

        # Disable RF Generator DLL test mode.
        self.VNXdll.fnLMS_SetTestMode(ctypes.c_bool(False)) # turn test mode off
        
        # Number of the currently connected devices.
        self._num_devs = 0

        
        # Create dictionaries that keeps track of last set power, frequency.
        self.LastFrequency = dict() 
        self.LastPower = dict()  
        # Dictionary to keep track of min/max powers and frequencies.
        self.MinMaxFrequency = dict()
        self.MinMaxPower = dict()
        # Create a dictionary that maps serial numbers to Device ID's.
        self.SerialNumberDict = dict()

        # Create a context for the server.
        self._pseudo_context = {}
        if self.autoRefresh:
            callLater(0.1, self.startRefreshing)
        else:
            self.refreshRFGenerators()

    def startRefreshing(self):
        """Start periodically refreshing the list of devices.

        The start call returns a deferred which we save for later.
        When the refresh loop is shutdown, we will wait for this
        deferred to fire to indicate that it has termin_attnted.
        """
        self.refresher = LoopingCall(self.refreshRFGenerators)
        self.refresherDone = self.refresher.start(self.refreshInterval, now=True)
        
    @inlineCallbacks
    def stopServer(self):
        """Kill the device refresh loop and wait for it to terminate."""
        if hasattr(self, 'refresher'):
            self.refresher.stop()
            yield self.refresherDone
        self.killRFGenConnections()
            
    def killRFGenConnections(self):
        try:    
            for DID in self.SerialNumberDict.itervalues():
                try:
                    self.VNXdll.fnLMS_CloseDevice(ctypes.c_uint(DID))
                except Exception:
                    pass
        except Exception:
            pass
    
    @inlineCallbacks
    def refreshRFGenerators(self):
        '''Refresh RF Generators list.'''
        n = yield self.VNXdll.fnLMS_GetNumDevices()
        if n == self._num_devs:
            pass
        elif n == 0:
            print('Lab Brick RF Generators disconnected')
            self._num_devs = n
            self.SerialNumberDict.clear() # why not clear self.MinMaxFrequency self.MinMaxPower dictionaries as well
            self.LastFrequency.clear()
            self.LastPower.clear()
        else:
            self._num_devs = n
            DEVIDs = (ctypes.c_uint * MAX_NUM_ATTEN)()
            DEVIDs_ptr = ctypes.cast(DEVIDs, ctypes.POINTER(ctypes.c_uint))
            yield self.VNXdll.fnLMS_GetDevInfo(DEVIDs_ptr)
            MODNAME = ctypes.create_string_buffer(MAX_MODEL_NAME)
            for idx in range(self._num_devs):
                SN = yield self.VNXdll.fnLMS_GetSerialNumber(DEVIDs_ptr[idx])
                self.SerialNumberDict.update({SN: DEVIDs_ptr[idx]})
                NameLength = yield self.VNXdll.fnLMS_GetModelName(DEVIDs_ptr[idx], MODNAME)
                self.select_rf_generator(self._pseudo_context, SN)
                freq = yield self.frequency(self._pseudo_context)
                power = yield self.power(self._pseudo_context)
                min_freq = yield self.min_frequency(self._pseudo_context)
                max_freq = yield self.max_frequency(self._pseudo_context)
                min_pow = yield self.min_power(self._pseudo_context)
                max_pow = yield self.max_power(self._pseudo_context)
                self.LastFrequency.update({SN: freq})
                self.LastPower.update({SN: power})
                self.MinMaxFrequency.update({SN: (min_freq, max_freq)})
                self.MinMaxPower.update({SN: (min_pow, max_pow)})
                print('Found a Lab Brick RF Generator with ' + MODNAME.raw[0:NameLength] + ', serial number: %i, current power: %.1f dB, current frequency %.1f GHz'%(SN, power, (freq/1e9)))

    def getDeviceDID(self, c):
        if 'SN' not in c:
            raise DeviceNotSelectedError("No Lab Brick RF Generator serial number selected")
        if c['SN'] not in self.SerialNumberDict.keys():
            raise Exception('Could not find Lab Brick RF Generator with serial number ' + c['SN'])
        return self.SerialNumberDict[c['SN']]       
                
    @setting(561, 'Refresh Device List')
    def refresh_device_list(self, c):
        '''Manually refresh RF Generator list.'''
        self.refreshRFGenerators
        
    @setting(562, 'List Devices', returns='*w')
    def list_devices(self, c):
        '''Return list of RF Generator serial numbers.'''
        return sorted(self.SerialNumberDict.keys())
        
    @setting(565, 'Select RF Generator', SN='w', returns='')
    def select_rf_generator(self, c, SN):
        '''Select RF Generator by its serial number. Since the serial numbers are unique by definition, no extra information is necessary to select a device.'''
        print "SN=", SN
        c['SN'] = SN
        
    @setting(566, 'Deselect RF Generator', returns='')
    def deselect_rf_generator(self, c):
        '''Deselect RF Generator.'''
        if 'SN' in c:
            del c['SN']
            
    @setting(532, 'Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def frequency(self, c, freq=None):
    	'''Set or get RF generator output frequency'''
    	DID = ctypes.c_uint(self.getDeviceDID(c))
    	yield self.VNXdll.fnLMS_InitDevice(DID)
    	if freq is None:
    		freq = 10. * (yield self.VNXdll.fnLMS_GetFrequency(DID)) * units.Hz # Synthesizer decided to work in 10 Hz increments f_actual = 10 * f_returned
    	else:
    		if self.LastFrequency[c['SN']] == freq:
    			return
    		if freq < self.MinMaxFrequency[c['SN']][0]:
    			freq = self.MinMaxFrequency[c['SN']][0]
    		if freq > self.MinMaxFrequency[c['SN']][1]:
    			freq = self.MinMaxFrequency[c['SN']][1]
    		self.LastFrequency[c['SN']] = freq
    		yield self.VNXdll.fnLMS_SetFrequency(DID, c_types.int(freq['Hz']*10))
    	yield self.VNXdll.fnLMS_CloseDevice(DID)
    	returnValue(freq)
    		
    @setting(545, 'Power', power=['v[dB]'], returns=['v[dB]'])		
    def power(self, c, power=None):
    	'''Set or get RF generator output power'''
    	DID = ctypes.c_uint(self.getDeviceDID(c))
    	yield self.VNXdll.fnLMS_InitDevice(DID)
    	if power is None:
                powerLev = yield self.VNXdll.fnLMS_GetPowerLevel(DID)
                #power = self.MinMaxPower[c['SN']][1] - (0.25*(yield self.VNXdll.fnLMS_GetPowerLevel(DID)))*units.dBm
                power = (0.25*(yield self.VNXdll.fnLMS_GetPowerLevel(DID)))*units.dBm
    	else:
    		if self.LastPower[c['SN']] == power:
    			return
    		if power < self.MinMaxPower[c['SN']][0]:
    			power = self.MinMaxPower[c['SN']][0]
    		if power > self.MinMaxPower[c['SN']][1]:
    			power = self.MinMaxPower[c['SN']][1]
    		self.LastPower[c['SN']] = power
    		powerSetting = c_types.int(4*(self.MinMaxPower[c['SN']][1] - power['dBm']))
    		yield self.VNXdll.fnLMS_SetFrequency(DID, powerSetting)
    	yield self.VNXdll.fnLMS_CloseDevice(DID)
    	returnValue(power)

    @setting(5521, 'Max Power', returns='v[dBm]')
    def max_power(self, c):
        '''Return maximum output power.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLMS_InitDevice(DID)
        max_pow = 0.25 * (yield self.VNXdll.fnLMS_GetMaxPwr(DID))
        yield self.VNXdll.fnLMS_CloseDevice(DID)
        returnValue(units.Value(max_pow, 'dBm'))

    @setting(5522, 'Min Power', returns='v[dBm]')
    def min_power(self, c):
        '''Return minimum output power.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLMS_InitDevice(DID)
        min_pow = 0.25 * (yield self.VNXdll.fnLMS_GetMinPwr(DID))
        yield self.VNXdll.fnLMS_CloseDevice(DID)
        returnValue(units.Value(min_pow, 'dBm'))
        
    @setting(5523, 'Max Frequency', returns='v[Hz]')
    def max_frequency(self, c):
        '''Return maximum output frequency.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLMS_InitDevice(DID)
        max_attn = 10. * (yield self.VNXdll.fnLMS_GetMaxFreq(DID))
        yield self.VNXdll.fnLMS_CloseDevice(DID)
        returnValue(units.Value(max_attn, 'Hz'))

    @setting(5524, 'Min Frequency', returns='v[Hz]')
    def min_frequency(self, c):
        '''Return minimum output frequency.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLMS_InitDevice(DID)
        min_attn = 10. * (yield self.VNXdll.fnLMS_GetMinFreq(DID))
        yield self.VNXdll.fnLMS_CloseDevice(DID)
        returnValue(units.Value(min_attn, 'Hz'))
        
    @setting(5530, 'Model Name', returns='s')
    def model_name(self, c):
        '''Return RF generator model name.'''
        MODNAME = ctypes.create_string_buffer(MAX_MODEL_NAME)
        NameLength = yield self.VNXdll.fnLMS_GetModelName(self.getDeviceDID(c), MODNAME)
        returnValue(''.join(MODNAME.raw[0:NameLength]))

__server__ = LBRFGenServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
