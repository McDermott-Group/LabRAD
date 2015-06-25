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
description =  
instancename = %LABRADNODE% LBA Server

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
MAX_MODELNAME = 32      # maximum length of LabBrick model name

class LBAttenuatorServer(LabradServer):
    name='%LABRADNODE% LBA Server'
    refreshInterval = 3
    defaultTimeout = 0.1
    
    @inlineCallbacks
    def GetRegistryKeys(self):
        '''Get registry keys for the Lab Brick Attenuator Server.'''
        reg = self.client.registry()
        yield reg.cd(['', 'Servers', 'Lab Brick Attenuators'], True)
        dirs, keys = yield reg.dir()
        if 'Lab Brick Attenuator DLL Path' not in keys:
            self.DLL_path = ''
        else:
            self.DLL_path = yield reg.get('Lab Brick Attenuator DLL Path')        
        if 'Lab Brick Attenuator Timeout' not in keys:
            self.waitTime = self.defaultTimeout
        else:
            self.waitTime = yield reg.get('Lab Brick Attenuator Timeout')
        if 'Lab Brick Attenuator Server Autorefresh' not in keys:
            self.autoRefresh = False
        else:
            self.autoRefresh = yield reg.get('Lab Brick Attenuator Server Autorefresh')
    
    @inlineCallbacks    
    def initServer(self):
        '''Initialize the Lab Brick Attenuator Server.'''
        yield self.GetRegistryKeys()
        try:
            self.VNXdll = ctypes.CDLL(self.DLL_path)
        except:
            raise Exception('Could not find Lab Brick Attenuator DLL')

        # Disable attenuator DLL test mode.
        TestMode = ctypes.c_bool(False)
        self.VNXdll.fnLDA_SetTestMode(TestMode) 
        
        self.ndevices = 0

        # Create a dictionary that maps serial numbers to Device ID's.
        self.SerialNumberDict = dict()
        # Create a dictionary that keeps track of last set attenuation.
        self.LastAttenuation = dict()   
        # Dictionary to keep track of min/max attenuations.
        self.MinMaxAttenuation = dict()
        
        if self.autoRefresh:
            callLater(0.1, self.startRefreshing)
        else:
            self.refreshAttenuators()

    def startRefreshing(self):
        """Start periodically refreshing the list of devices.

        The start call returns a deferred which we save for later.
        When the refresh loop is shutdown, we will wait for this
        deferred to fire to indicate that it has terminated.
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
        for did in self.SerialNumberDict.itervalues():
            try:
                self.VNXdll.fnLDA_CloseDevice(ctypes.c_uint(did))
            except:
                pass
    
    @inlineCallbacks
    def refreshAttenuators(self):
        '''Update attenuator list.'''
        n = self.VNXdll.fnLDA_GetNumDevices()
        if n == self.ndevices:
            pass
        elif n == 0:
            print('Lab Brick attenuators disconnected.')
            self.ndevices = n
            self.SerialNumberDict.clear()
            self.LastAttenuation.clear()
        else:   
            self.ndevices = n        
            DEVIDs = (ctypes.c_uint * MAX_NUM_ATTEN)()
            DEVIDs_ptr = ctypes.cast(DEVIDs, ctypes.POINTER(ctypes.c_uint))
            self.VNXdll.fnLDA_GetDevInfo(DEVIDs_ptr)    
            MODNAME = ctypes.create_string_buffer(MAX_MODELNAME)
            for idx in range(self.ndevices):
                SN = self.VNXdll.fnLDA_GetSerialNumber(DEVIDs_ptr[idx])
                self.SerialNumberDict.update({SN:DEVIDs_ptr[idx]})
                nameLength = self.VNXdll.fnLDA_GetModelName(DEVIDs_ptr[idx], MODNAME)
                aDB = yield self.GetAttenuation(0, SN)
                aMin = yield self.MinAttenuation(0, SN)
                aMax = yield self.MaxAttenuation(0, SN)
                self.LastAttenuation.update({SN: aDB})
                self.MinMaxAttenuation.update({SN: (aMin, aMax)})
                print('Found a Lab Brick Attenuator with ' + MODNAME.raw[0:nameLength] + ', serial number: %i, current attenuation: %.1f dB'%(SN, aDB))

    def getDevice(self, c):
        if 'SN' not in c:
            raise DeviceNotSelectedError("No Lab Brick Attenuator serial number selected")
        if c['SN'] not in self.SerialNumberDict.keys():
            raise Exception('Could not find Lab Brick Attenuator with serial number ' + c['SN'])
        return self.SerialNumberDict[c['SN']]       
                
    @setting(1, 'Refresh Device List')
    def RefreshDeviceList(self, c):
        '''Manually refresh attenuator list.'''
        self.refreshAttenuators
        
    @setting(2, 'List Devices', returns='*i')
    def ListDevices(self, c):
        '''Return list of attenuator serial numbers'.'''
        return sorted(self.SerialNumberDict.keys())
        
    @setting(5, 'Select Attenuator', SN=['i', 'w'], returns='')
    def SelectAttenuator(self, c):
        '''Select attenuator.'''
        c['SN'] = SN
        
    @setting(6, 'Deselect Attenuator', returns='')
    def DeselectAttenuator(self, c):
        '''Deselect attenuator.'''
        if 'SN' in c:
            del c['SN']
     
    @setting(11, 'Get Attenuation', returns='v[dB]')
    def GetAttenuation(self, c):
        '''Get attenuation.'''
        DID = ctypes.c_uint(self.getDevice(c))
        self.VNXdll.fnLDA_InitDevice(DID)
        atten = 0.25 * self.VNXdll.fnLDA_GetAttenuation(DID)
        self.VNXdll.fnLDA_CloseDevice(DID)
        return atten * units.dB
        
    @setting(12, 'Set Attenuation', atten='v[dB]')
    def SetAttenuation(self, c, atten):
        '''Set attenuation.'''
        SN = self.getDevice(c)
        if atten < self.MinMaxAttenuation[SN][0]:
            atten = 0. * units.dB
        elif atten > self.MinMaxAttenuation[SN][1]:
            atten = 63. * units.dB
        # Check to make sure it needs to be changed.
        if self.LastAttenuation[SN] == atten:
            return
        self.LastAttenuation[SN] = atten
        DID = ctypes.c_uint(self.getDevice(c))
        self.VNXdll.fnLDA_InitDevice(DID)
        self.VNXdll.fnLDA_SetAttenuation(DID, ctypes.c_int(int(4.*atten)))
        self.VNXdll.fnLDA_CloseDevice(DID)
        
    @setting(21, 'Max Attenuation', returns='v[dB]')
    def MaxAttenuation(self, c):
        '''Return maximum attenuation.'''
        DID = ctypes.c_uint(self.getDevice(c))
        self.VNXdll.fnLDA_InitDevice(DID)
        maxA = self.VNXdll.fnLDA_GetMaxAttenuation(DID)
        self.VNXdll.fnLDA_CloseDevice(DID)
        return maxA * untis.dB

    @setting(22, 'Min Attenuation', returns='v[dB]')
    def MinAttenuation(self, c):
        '''Return minimum attenuation.'''
        DID = ctypes.c_uint(self.getDevice(c))
        self.VNXdll.fnLDA_InitDevice(DID)
        minA = self.VNXdll.fnLDA_GetMinAttenuation(DID)
        self.VNXdll.fnLDA_CloseDevice(DID)
        return minA * units.dB
        
    @setting(30, 'Model Name', returns='s')
    def ModelName(self, c):
        '''Return attenuator model name.'''
        MODNAME = ctypes.create_string_buffer(MAX_MODELNAME)
        nameLength = self.VNXdll.fnLDA_GetModelName(self.getDevice(c), MODNAME)
        return  ''.join(MODNAME.raw[0:nameLength])

__server__ = LBAttenuatorServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)