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
### BEGIN NODE INFO
[info]
name = RF Generators
version = 0.1
description = This server provides basic control for microwave generators.

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
from labrad.units import Hz, dBm
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper


class HP83712BWrapper(GPIBDeviceWrapper):
    @inlineCallbacks
    def initialize(self):
        self.frequency = yield self.getFrequency()
        self.power = yield self.getPower()
        self.output =  yield self.getOutput()
    
    @inlineCallbacks    
    def reset(self):
        yield self.write('*CLS;*RST')
        yield self.initialize()

    @inlineCallbacks
    def getFrequency(self):
        self.frequency = (yield self.query('FREQ?').addCallback(float)) * Hz
        returnValue(self.frequency)

    @inlineCallbacks
    def getPower(self):
        self.power = (yield self.query('POW?').addCallback(float)) * dBm
        returnValue(self.power)
    
    @inlineCallbacks
    def getOutput(self):
        self.output = yield self.query('OUTP?').addCallback(float).addCallback(bool)
        returnValue(self.output)

    @inlineCallbacks
    def setFrequency(self, freq):
        if self.frequency != freq:
            yield self.write('FREQ ' + str(freq['Hz']))
            self.frequency = freq
    
    @inlineCallbacks
    def setPower(self, pow):
        if self.power != pow:
            yield self.write('POW ' + str(pow['dBm']))
            self.power = pow

    @inlineCallbacks
    def setOutput(self, out):
        if self.output != bool(out):
            yield self.write('OUTP ' + str(int(out)))
            self.output = bool(out)


class HP83620AWrapper(HP83712BWrapper):
    @inlineCallbacks
    def setOutput(self, out):
        if self.output != bool(out):
            yield self.write('POW:STAT ' + str(int(out)))
            self.output = bool(out)


class HP8341BWrapper(GPIBDeviceWrapper):
    @inlineCallbacks
    def initialize(self):
        self.frequency = yield self.getFrequency()
        self.power = yield self.getPower()
        self.output =  yield self.getOutput()
    
    @inlineCallbacks    
    def reset(self):
        yield self.write('CS')
        yield self.initialize()

    @inlineCallbacks
    def getFrequency(self):
        self.frequency = (yield self.query('OK').addCallback(float)) * Hz
        returnValue(self.frequency)

    @inlineCallbacks
    def getPower(self):
        self.power = (yield self.query('OR').addCallback(float)) * dBm
        returnValue(self.power)
    
    @inlineCallbacks
    def getOutput(self):
        yield self.write('OM')
        status = yield self.read_raw() 
        self.output = bool(ord(status[6]) & 32);
        returnValue(self.output)

    @inlineCallbacks
    def setFrequency(self, freq):
        if self.frequency != freq:
            yield self.write('CW' + str(freq['GHz']) + 'GZ')
            self.frequency = freq
    
    @inlineCallbacks
    def setPower(self, pow):
        if self.power != pow:
            yield self.write('PL' + str(pow['dBm']) + 'DB')
            self.power = pow

    @inlineCallbacks
    def setOutput(self, out):
        if self.output != bool(out):
            yield self.write('RF' + str(int(out)))
            self.output = bool(out)

            
class RFGeneratorServer(GPIBManagedServer):
    """This server provides basic control for microwave generators."""
    name = 'RF Generators'
    deviceWrappers={'HEWLETT-PACKARD 83620A': HP83620AWrapper,      # HP83620A hasn't yet been tested.
                    'HEWLETT-PACKARD 83712B': HP83712BWrapper,
                    'HEWLETT-PACKARD 8341B':  HP8341BWrapper}

    @setting(9, 'Reset')
    def reset(self, c):
        """Reset the RF generator."""
        yield self.selectedDevice(c).reset()

    @setting(10, 'Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def frequency(self, c, freq=None):
        """Get or set the CW frequency."""
        dev = self.selectedDevice(c)
        if freq is not None:
            yield dev.setFrequency(freq)
        returnValue(dev.frequency)

    @setting(11, 'Power', pow=['v[dBm]'], returns=['v[dBm]'])
    def power(self, c, pow=None):
        """Get or set the CW power."""
        dev = self.selectedDevice(c)
        if pow is not None:
            yield dev.setPower(pow)
        returnValue(dev.power)

    @setting(12, 'Output', out=['b'], returns=['b'])
    def output(self, c, out=None):
        """Get or set the RF generator output."""
        dev = self.selectedDevice(c)
        if out is not None:
            yield dev.setOutput(out)
        returnValue(dev.output)
        
__server__ = RFGeneratorServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)