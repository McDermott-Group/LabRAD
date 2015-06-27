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

"""
### BEGIN NODE INFO
[info]
name = Lab Brick Attenuator Server
version = 1.1
description =  Gives access to Lab Brick attenuators. This server self-refreshes.
instancename = %LABRADNODE% LBA

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

MAX_NUM_ATTEN = 64      # maximum number of connected attenuators
MAX_MODEL_NAME = 32     # maximum length of Lab Brick model name

class LBAttenuatorServer(LabradServer):
    name='%LABRADNODE% LBA'
    refreshInterval = 3
    defaultTimeout = 0.1 * units.s
    
    @inlineCallbacks
    def getRegistryKeys(self):
        '''Get registry keys for the Lab Brick Attenuator Server.'''
        reg = self.client.registry()
        yield reg.cd(['', 'Servers', 'Lab Brick Attenuators'], True)
        dirs, keys = yield reg.dir()
        if 'Lab Brick Attenuator DLL Path' not in keys:
            self.DLL_path = 'Z:\mcdermott-group\LabRAD\LabBricks\VNX_atten.dll'
        else:
            self.DLL_path = yield reg.get('Lab Brick Attenuator DLL Path')
        print("Lab Brick Attenuator DLL Path is set to " + str(self.DLL_path))
        if 'Lab Brick Attenuator Timeout' not in keys:
            self.waitTime = self.defaultTimeout
        else:
            self.waitTime = yield reg.get('Lab Brick Attenuator Timeout')
        print("Lab Brick Attenuator Timeout is set to " + str(self.waitTime))
        if 'Lab Brick Attenuator Server Autorefresh' not in keys:
            self.autoRefresh = False
        else:
            self.autoRefresh = yield reg.get('Lab Brick Attenuator Server Autorefresh')
        print("Lab Brick Attenuator Server Autorefresh is set to " + str(self.autoRefresh))

    @inlineCallbacks    
    def initServer(self):
        '''Initialize the Lab Brick Attenuator Server.'''
        yield self.getRegistryKeys()
        try:
            self.VNXdll = yield ctypes.CDLL(self.DLL_path)
        except Exception:
            raise Exception('Could not find Lab Brick Attenuator DLL')

        # Disable attenuator DLL test mode.
        self.VNXdll.fnLDA_SetTestMode(ctypes.c_bool(False)) 
        
        # Number of the currently connected devices.
        self._num_devs = 0

        # Create a dictionary that maps serial numbers to Device ID's.
        self.SerialNumberDict = dict()
        # Create a dictionary that keeps track of last set attenuation.
        self.LastAttenuation = dict()   
        # Dictionary to keep track of min/max attenuations.
        self.MinMaxAttenuation = dict()

        # Create a context for the server.
        self._pseudo_context = {}
        if self.autoRefresh:
            callLater(0.1, self.startRefreshing)
        else:
            self.refreshAttenuators()

    def startRefreshing(self):
        """Start periodically refreshing the list of devices.

        The start call returns a deferred which we save for later.
        When the refresh loop is shutdown, we will wait for this
        deferred to fire to indicate that it has termin_attnted.
        """
        self.refresher = LoopingCall(self.refreshAttenuators)
        self.refresherDone = self.refresher.start(self.refreshInterval, now=True)
        
    @inlineCallbacks
    def stopServer(self):
        """Kill the device refresh loop and wait for it to terminate."""
        if hasattr(self, 'refresher'):
            self.refresher.stop()
            yield self.refresherDone
        self.killAttenuatorConnections()
            
    def killAttenuatorConnections(self):
        for DID in self.SerialNumberDict.itervalues():
            try:
                self.VNXdll.fnLDA_CloseDevice(ctypes.c_uint(DID))
            except Exception:
                pass
    
    @inlineCallbacks
    def refreshAttenuators(self):
        '''Refresh attenuator list.'''
        n = yield self.VNXdll.fnLDA_GetNumDevices()
        if n == self._num_devs:
            pass
        elif n == 0:
            print('Lab Brick attenuators disconnected')
            self._num_devs = n
            self.SerialNumberDict.clear()
            self.LastAttenuation.clear()
        else:
            self._num_devs = n
            DEVIDs = (ctypes.c_uint * MAX_NUM_ATTEN)()
            DEVIDs_ptr = ctypes.cast(DEVIDs, ctypes.POINTER(ctypes.c_uint))
            yield self.VNXdll.fnLDA_GetDevInfo(DEVIDs_ptr)
            MODNAME = ctypes.create_string_buffer(MAX_MODEL_NAME)
            for idx in range(self._num_devs):
                SN = yield self.VNXdll.fnLDA_GetSerialNumber(DEVIDs_ptr[idx])
                self.SerialNumberDict.update({SN: DEVIDs_ptr[idx]})
                NameLength = yield self.VNXdll.fnLDA_GetModelName(DEVIDs_ptr[idx], MODNAME)
                self.select_attenuator(self._pseudo_context, SN)
                attn_dB = yield self.get_attenuation(self._pseudo_context)
                min_attn = yield self.min_attenuation(self._pseudo_context)
                max_attn = yield self.max_attenuation(self._pseudo_context)
                self.LastAttenuation.update({SN: attn_dB})
                self.MinMaxAttenuation.update({SN: (min_attn, max_attn)})
                print('Found a Lab Brick Attenuator with ' + MODNAME.raw[0:NameLength] + ', serial number: %i, current attenuation: %.1f dB'%(SN, attn_dB))

    def getDeviceDID(self, c):
        if 'SN' not in c:
            raise DeviceNotSelectedError("No Lab Brick Attenuator serial number selected")
        if c['SN'] not in self.SerialNumberDict.keys():
            raise Exception('Could not find Lab Brick Attenuator with serial number ' + c['SN'])
        return self.SerialNumberDict[c['SN']]       
                
    @setting(1, 'Refresh Device List')
    def refresh_device_list(self, c):
        '''Manually refresh attenuator list.'''
        self.refreshAttenuators
        
    @setting(2, 'List Devices', returns='*w')
    def list_devices(self, c):
        '''Return list of attenuator serial numbers.'''
        return sorted(self.SerialNumberDict.keys())
        
    @setting(5, 'Select Attenuator', SN='w', returns='')
    def select_attenuator(self, c, SN):
        '''Select attenuator by its serial number. Since the serial numbers are unique by definition, no extra information is necessary to select a device.'''
        c['SN'] = SN
        
    @setting(6, 'Deselect Attenuator', returns='')
    def deselect_attenuator(self, c):
        '''Deselect attenuator.'''
        if 'SN' in c:
            del c['SN']
     
    @setting(11, 'Get Attenuation', returns='v[dB]')
    def get_attenuation(self, c):
        '''Get attenuation.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLDA_InitDevice(DID)
        atten = 0.25 * (yield self.VNXdll.fnLDA_GetAttenuation(DID))
        yield self.VNXdll.fnLDA_CloseDevice(DID)
        returnValue(units.Value(atten, 'dB'))
        
    @setting(12, 'Set Attenuation', atten='v[dB]')
    def set_attenuation(self, c, atten):
        '''Set attenuation.'''
        SN = self.getDeviceDID(c)
        if atten < self.MinMaxAttenuation[c['SN']][0]:
            atten = self.MinMaxAttenuation[c['SN']][0]
        elif atten > self.MinMaxAttenuation[c['SN']][1]:
            atten = self.MinMaxAttenuation[c['SN']][1]
        # Check to make sure it needs to be changed.
        if self.LastAttenuation[c['SN']] == atten:
            return
        self.LastAttenuation[c['SN']] = atten
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLDA_InitDevice(DID)
        yield self.VNXdll.fnLDA_SetAttenuation(DID, ctypes.c_int(int(4. * atten)))
        yield self.VNXdll.fnLDA_CloseDevice(DID)
        
    @setting(21, 'Max Attenuation', returns='v[dB]')
    def max_attenuation(self, c):
        '''Return maximum attenuation.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLDA_InitDevice(DID)
        max_attn = 0.25 * (yield self.VNXdll.fnLDA_GetMaxAttenuation(DID))
        yield self.VNXdll.fnLDA_CloseDevice(DID)
        returnValue(units.Value(max_attn, 'dB'))

    @setting(22, 'Min Attenuation', returns='v[dB]')
    def min_attenuation(self, c):
        '''Return minimum attenuation.'''
        DID = ctypes.c_uint(self.getDeviceDID(c))
        yield self.VNXdll.fnLDA_InitDevice(DID)
        min_attn = 0.25 * (yield self.VNXdll.fnLDA_GetMinAttenuation(DID))
        yield self.VNXdll.fnLDA_CloseDevice(DID)
        returnValue(units.Value(min_attn, 'dB'))
        
    @setting(30, 'Model Name', returns='s')
    def model_name(self, c):
        '''Return attenuator model name.'''
        MODNAME = ctypes.create_string_buffer(MAX_MODEL_NAME)
        NameLength = yield self.VNXdll.fnLDA_GetModelName(self.getDeviceDID(c), MODNAME)
        returnValue(''.join(MODNAME.raw[0:NameLength]))

__server__ = LBAttenuatorServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)