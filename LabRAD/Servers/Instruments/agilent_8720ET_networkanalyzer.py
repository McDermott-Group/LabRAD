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
name = Agilent 8720ET NA
version = 0.1
description = Two channel 8720ET transmission/reflection network analyzer 

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from labrad.server import setting, returnValue
import labrad.units as units
import time, numpy

class Agilent8720ETServer(GPIBManagedServer):
    name = 'Agilent 8720ET NA'
    deviceName = 'HEWLETT PACKARD 8720ET'
    deviceWrapper = GPIBDeviceWrapper
    
    @setting(431, 'Preset NA')
    def preset(self, c):
        """Reset the NA."""
        dev = self.selectedDevice(c)
        yield dev.query('OPC?;PRES;')
        
    @setting(432, 'Start Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def startFrequency(self, c, freq=None):
        '''Set or get the start frequency of the sweep'''
        dev = self.selectedDevice(c)
        if freq is None:
            resp = yield dev.query('STAR?;')
            freq = float(resp)*units.Hz
        else:
            yield dev.write('STAR%i;' % freq['Hz'])
        
        returnValue(freq)
    
    @setting(433, 'Stop Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def stopFrequency(self, c, freq=None):
        '''Set or get the stop frequency of the sweep'''
        dev = self.selectedDevice(c)
        if freq is None:
            resp = yield dev.query('STOP?;')
            freq = float(resp)*units.Hz
        else:
            yield dev.write('STOP%i;' % freq['Hz'])
        
        returnValue(freq)
    
    @setting(444, 'Power', pow=['v[dBm]'], returns=['v[dBm]'])
    def power(self, c, pow=None):
        '''Set or get the sweep power level'''
        dev = self.selectedDevice(c)
        if pow is None:
            resp = yield dev.query('POWE?;')
            pow = float(resp)*units.dBm
        else:
            if pow['dBm'] < -10:
                print 'Minimum power level for Agilent 8720ET is -10 dBm.'
                pow = -10*units.dBm;
            if pow['dBm'] > 10:
                print 'Maximum power level for Agilent 8720ER is 10 dBm.'
                pow = 10*units.dBm;
            yield dev.write('POWE%iDB;' % pow['dBm'])
        returnValue(pow)
        
    @setting(435, 'Measurement Type', mode=['s'], returns=['s'])
    def measType(self, c, mode=None):
        '''Set or get the measurement mode [t]ransmission or [r]eflection'''
        dev = self.selectedDevice(c)
        
        if mode is None:
            resp = yield dev.query('RFLP?;')
            if bool(int(resp)):
                mode = 'r'
            else:
                mode = 't'
        else:
            if mode == 'r':
                yield dev.write('RFLP;')
            if mode == 't':
                yield dev.write('TRAP;')
            else:
                print 'Unknown mode '+mode
        
        returnValue(mode)
        
    @setting(436, 'Number of Points', pn=['i'], returns=['i'])
    def points(self, c, pn=None):
        '''Set or get number of points in a sweep'''
        dev = self.selectedDevice(c)
        
        if pn is None:
            resp = yield dev.query('POIN?;')
            pn = int(float(resp))
        
        else:
            yield dev.write('POIN%i;'%pn)
            t = yield dev.query('SWET?;')
            wt = float(t)
            time.sleep(2.*wt) #be sure to wait for two sweep times as required in 8270ET docs
        
        returnValue(pn)   
    
    @setting(437, 'Average On Off', avg=['b'], returns=['b'])
    def averageOnOff(self, c, avg=None):
        '''Set or get average state'''
        dev = self.selectedDevice(c)
        if avg is None:
            resp = yield dev.query('AVERO?;')
            avg = bool(int(resp))
        else:
            if avg:
                yield dev.write('AVEROON;')
            else:
                yield dev.write('AVEROOFF;')
        returnValue(avg)
        
    @setting(438, 'Average Points', aN=['i'], returns=['i'])
    def averagePoints(self, c, aN=None):
        '''Set or get number of points in average'''
        dev = self.selectedDevice(c)
        if aN is None:
            N = yield dev.query('AVERFACT?;')
            aN = int(float(N))
        else:
            yield dev.write('AVERFACT%i;'%aN)
        returnValue(aN)
        
    @setting(439, 'Restart Averaging')
    def restartAverage(self, c):
        '''Restart trace averaging'''
        dev = self.selectedDevice(c)
        yield dev.write('AVERREST;')
        
    @setting(4310, 'Get Trace', returns=['*c'])
    def getTrace(self, c):
        '''Get trace from NA'''
        dev = self.selectedDevice(c)
        
        yield dev.write('FORM5;')
        
        numAvg = yield self.averagePoints(c)
        print numAvg
        waitTime = yield dev.query('SWET?;')
        print waitTime
        sweepWait = numAvg*float(waitTime) + 0.1
        
        yield self.restartAverage(c)
        time.sleep(sweepWait)
        
        yield dev.write('OUTPFORM;')
        dataBuffer = yield dev.read_raw()
        print dataBuffer
        rawData = numpy.fromstring(dataBuffer,dtype=numpy.float32)
        
        data =  rawData[1:2:-1] + 1j*rawData[2:2:-1] 
        
        returnValue(data)
        
        
        
            
        
    
    @setting(99, 'Initialize NA')
    def initialize(self, c):
        """Initialize the NA"""
        dev = self.selectedDevice(c)
        
        yield self.preset(c)
        yield dev.write('CHAN1;')
        yield dev.write('AUXCOFF;')
        yield dev.write('S21;')
        yield dev.write('LOGM;AUTO;')
        yield self.points(c, 801)
        
        print 'Initialized...'
        
        
       
        
        
        

__server__ = Agilent8720ETServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
