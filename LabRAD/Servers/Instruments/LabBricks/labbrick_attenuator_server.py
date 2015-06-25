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
name = Lab Brick Attenuator Server
version = 1.0
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

from labrad import types as T, util
from labrad.errors import Error
from labrad.server import LabradServer, setting

from twisted.internet.defer import inlineCallbacks, returnValue, gatherResults
from twisted.internet.reactor import callLater
from twisted.internet.task import LoopingCall

import time
import ctypes

MAX_NUM_ATTEN = 64      # maximum number of connected attenuators
MAX_MODELNAME = 32      # maximum size of LABBRICK model name


#the server itself...
class LBAttenuatorServer(LabradServer):

    name='%LABRADNODE% LBA Server'
    
    refreshInterval = 3
    defaultTimeout = 0.05
    
    @inlineCallbacks
    def GetRegistryKeys(self):
        '''get registry keys for this server'''
        
        reg = self.client.registry()
        
        yield reg.cd(['', 'Servers', 'Lab Brick Attenuators'], True)
        
        dirs,keys = yield reg.dir()
                
        self.DLL_path = yield reg.get('Lab Brick Attenuator DLL Path')
                
        if 'Lab Brick Attenuator Timeout' not in keys:
            self.WaitTime = self.defaultTimeout
        else:
            self.WaitTime = yield reg.get('Lab Brick Attenuator Timeout')
        
        if 'Lab Brick Attenuator Server Autorefresh' not in keys:
            self.autorefresh = False
        else:
            self.autorefresh = yield reg.get('Lab Brick Attenuator Server Autorefresh')
    
    @inlineCallbacks    
    def initServer(self):

        self.DLL_path = ''
        self.WaitTime = self.defaultTimeout
        self.autorefresh = True
        
        yield self.GetRegistryKeys()
                
        try:
            self.VNXdll = ctypes.CDLL(self.DLL_path)
        except:
            raise Exception('Could not find Lab Brick Attenuator DLL')
            
        
        #disable attenuator DLL test mode
        TestMode = ctypes.c_bool(False)
        self.VNXdll.fnLDA_SetTestMode(TestMode) 
        
        self.ndevices = 0

        #create a dictionary that maps serial numbers to Device ID's
        self.SerialNumberDict = dict()
        #create a dictionary that keeps track of last set attenuation
        self.LastAttenuation = dict()   
        #dictionary to keep track of min/max attenuations
        self.MinMaxAttenuation = dict()
        
        if self.autorefresh:
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
        try:
            for did in self.SerialNumberDict.itervalues():
                self.VNXdll.fnLDA_CloseDevice(ctypes.c_uint(did))
        except:
            pass
    
    @inlineCallbacks
    def refreshAttenuators(self):
        '''Update attenuator list'''
        
        n = self.VNXdll.fnLDA_GetNumDevices()
                        
        if self.ndevices == n:
            return
            
        elif n == 0:
            print 'Attenuators disconnected.'
            self.ndevices = n
            self.SerialNumberDict.clear()
            self.LastAttenuation.clear()
        
        else:   

            self.ndevices = n        
            
            DEVIDs = (ctypes.c_uint*MAX_NUM_ATTEN)()
            DEVIDs_ptr = ctypes.cast(DEVIDs, ctypes.POINTER(ctypes.c_uint))
            
            self.VNXdll.fnLDA_GetDevInfo(DEVIDs_ptr)    
            MODNAME = ctypes.create_string_buffer(MAX_MODELNAME)
            
            print 'New Lab Brick attenuators found: '
            for idx in range(self.ndevices):
                
                SN = self.VNXdll.fnLDA_GetSerialNumber(DEVIDs_ptr[idx])
                self.SerialNumberDict.update({SN:DEVIDs_ptr[idx]})
                
                nameLength = self.VNXdll.fnLDA_GetModelName(DEVIDs_ptr[idx], MODNAME)
                
                aDB = yield self.GetAttenuation(0,SN)
                aMin = yield self.MinAttenuation(0,SN)
                aMax = yield self.MaxAttenuation(0,SN)
                
                self.LastAttenuation.update({SN:aDB})
                self.MinMaxAttenuation.update({SN:(aMin,aMax)})
                
                print MODNAME.raw[0:nameLength] + ' Serial Number: %i, Current attenuation: %.1f dB'%(SN,aDB)
            
                
    @setting(1)
    def update_attenuators(self, c):
        '''manually refresh attenuator list'''
        self.refreshAttenuators
        
    @setting(2, returns='*i')
    def list_devices(self,c):
        '''return list of attenuator serial numbers'''
        return sorted(self.SerialNumberDict.keys())
    
            
    @setting(11, 'Get Attenuation', SN=['i','w'], returns='v[dB]')
    def GetAttenuation(self, c, SN):
        '''Gets attenuation (in dB) from attenuator with given serial number'''
        
        if SN not in self.SerialNumberDict.keys():
            raise Exception('Attenuator with serial number %d is not connected to this PC!'%SN)
        
        DID = ctypes.c_uint(self.SerialNumberDict[SN])
        
        
        self.VNXdll.fnLDA_InitDevice(DID)
    
        atten = 0.25*self.VNXdll.fnLDA_GetAttenuation(DID)
    
        self.VNXdll.fnLDA_CloseDevice(DID)
            
        time.sleep(self.WaitTime)
        
        return atten
        
    @setting(12, 'Set Attenuation', SN=['i', 'w'], atten='v[dB]')
    def SetAttenuation(self, c, SN, atten):
        '''sets the attenuation of a given Serial Number'''
        
        if SN not in self.SerialNumberDict.keys():
            raise Exception('Attenuator with serial number %d is not connected to this PC!'%SN)
        
        if atten < self.MinMaxAttenuation[SN][0]:
            atten = 0.
        elif atten > self.MinMaxAttenuation[SN][1]:
            atten = 63.
        
        #check to make sure it needs to be changed
        if self.LastAttenuation[SN] == atten:
            return
        
        self.LastAttenuation[SN] = atten
        
        DID = ctypes.c_uint(self.SerialNumberDict[SN])
        
        self.VNXdll.fnLDA_InitDevice(DID)
        
        self.VNXdll.fnLDA_SetAttenuation(DID, ctypes.c_int(int(atten*4.)))
        
        self.VNXdll.fnLDA_CloseDevice(DID)
        
        time.sleep(self.WaitTime)
        
    @setting(21, 'Max Attenuation', SN=['i','w'], returns='v[dB]')
    def MaxAttenuation(self, c, SN):
        '''return maximum attenuation available to this attenuator'''
        
        if SN not in self.SerialNumberDict.keys():
            raise Exception('Attenuator with serial number %d is not connected to this PC!'%SN)
            
        DID = ctypes.c_uint(self.SerialNumberDict[SN])
        
        self.VNXdll.fnLDA_InitDevice(DID)
        
        maxA = self.VNXdll.fnLDA_GetMaxAttenuation(DID)
        
        self.VNXdll.fnLDA_CloseDevice(DID)
        
        return maxA
        
        
        
    @setting(22, 'Min Attenuation', SN=['i','w'], returns='v[dB]')
    def MinAttenuation(self, c, SN):
        '''return maximum attenuation available to this attenuator'''
        
        if SN not in self.SerialNumberDict.keys():
            raise Exception('Attenuator with serial number %d is not connected to this PC!'%SN)
            
        DID = ctypes.c_uint(self.SerialNumberDict[SN])
        
        self.VNXdll.fnLDA_InitDevice(DID)
        
        minA = self.VNXdll.fnLDA_GetMinAttenuation(DID)
        
        self.VNXdll.fnLDA_CloseDevice(DID)
        
        return minA
        
    @setting(30, 'Model Name', SN=['i','w'], returns='s')
    def ModelName(self,c, SN):
        ''' return attenuator model name '''
        if SN not in self.SerialNumberDict.keys():
            raise Exception('Attenuator with serial number %d is not connected to this PC!'%SN)
            
        MODNAME = ctypes.create_string_buffer(MAX_MODELNAME)
        nameLength = self.VNXdll.fnLDA_GetModelName(self.SerialNumberDict[SN], MODNAME)
        return  ''.join(MODNAME.raw[0:nameLength])
        
            
            
__server__ = LBAttenuatorServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
            
    
        
    
        