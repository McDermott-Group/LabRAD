# Copyright (C) 2015 Joseph Suttle
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
name = CSA7404B
version = 0.0.1
description = Server interface for CSA7404B Oscilloscope.
instancename = CSA7404B

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

from twisted.internet.defer import inlineCallbacks, returnValue

from labrad.server import setting
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from labrad.units import V, GHz,Hz


class CSA7404BWrapper(GPIBDeviceWrapper):
    """@inlineCallbacks
    def initialize(self):
        self.voltage = yield self.getVoltage()
        self.output =  yield self.getOutput()
    
    @inlineCallbacks    
    def reset(self):
        yield self.write('*CLS;*RST')
        yield self.initialize()

    @inlineCallbacks
    def getVoltage(self):
        self.voltage = (yield self.query('VOLT?').addCallback(float)) * V
        returnValue(self.voltage)
    
    @inlineCallbacks
    def getOutput(self):
        self.output = yield self.query('EXON?').addCallback(int).addCallback(bool)
        returnValue(self.output)

    @inlineCallbacks
    def setVoltage(self, v):
        if self.voltage != v:
            yield self.write('VOLT ' + str(v['V']))
            # Ensure that the voltage is actually set to the right level.
            self.voltage = yield self.getVoltage()

    @inlineCallbacks
    def setOutput(self, on):
        if self.output != bool(on):
            yield self.write('EXON ' + str(int(on)))
            # Ensure that the output is set properly.
            self.output = yield self.getOutput()"""
    
##Basic System Functionality
    @inlineCallbacks
    def initialize(self):
        self.Mode = yield self.getAcqMode()
        self.SampleRate =  yield self.getSampleRate()
        print('Initialized')
    
    @inlineCallbacks    
    def reset(self):
        yield self.write('*CLS;*RST')
        yield self.initialize()
        print('Device Reset')
            

##Acquisition Commands
    ##Acquisition Mode Commands
    @inlineCallbacks
    def setAcqMode(self,mode):
        #Mode should be a string denoting what acqusition mode you'd like to use.
        #Options are sample, peakdetect, hires, average, envelope, wfmdb
        #Not all modes are fully supported at this time and may require some fiddling.
        self.write('ACQUIRE:MODE '+str(mode))
        self.Mode = yield self.getAcqMode()
    @inlineCallbacks
    def getAcqMode(self):
        self.Mode = yield self.query('ACQUIRE:MODE?').addCallback(str)
        returnValue(self.Mode)
    
    @inlineCallbacks
    def getSampleMode(self):
    #Sample mode will tell you if the time base is interpolated or real time or something else 
        self.SampleMode = yield self.query('ACQ:SAMP?').addCallback(str)
        returnValue(self.SampleMode)
        
    @inlineCallbacks
    def getSampleRate(self):
    #Returns the current sample rate of the scope in samples per second.
        self.SampleRate = yield self.query('HOR:MAI:SAMPLER?')
        returnValue(self.SampleRate)
    @inlineCallbacks    
    def setSampleRate(self,samRate):
        #Mode should be a string denoting what acqusition mode you'd like to use.
        #Options are sample, peakdetect, hires, average, envelope, wfmdb
        #Not all modes are fully supported at this time and may require some fiddling.
        self.write('HOR:MAI:SAMPLER '+str(samRate))
        self.SampleRate = yield self.getSampleRate()


    
class CSA7404BServer(GPIBManagedServer):
    """Provides basic control for Tektronix CSA7404B Oscilloscope."""
    name = 'CSA7404B'
    deviceName = 'TEKTRONIX CSA7404B'
    deviceWrapper = CSA7404BWrapper
    
    @setting(100, 'Reset')
    def reset(self, c):
        """Reset the scope."""
        yield self.selectedDevice(c).reset()
        
    @setting(101, 'Sampling Rate', SmRt = ['v[Hz]'])
    def sampRate(self, c, SmRt=None):
        dev = self.selectedDevice(c)
        if SmRt is not None:
            yield dev.setSampleRate(SmRt)
        returnValue(dev.SampleRate)
    """
    @setting(101, 'Voltage', v=['v[V]'], returns=['v[V]'])
    def voltage(self, c, v=None):
        dev = self.selectedDevice(c)
        if v is not None:
            yield dev.setVoltage(v)
        returnValue(dev.voltage)
        
    @setting(102, 'Output', on=['b'], returns=['b'])
    def output(self, c, on=None):
        dev = self.selectedDevice(c)
        if on is not None:
            yield dev.setOutput(on)
        returnValue(dev.output)
    """

__server__ = CSA7404BServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)