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
version = 1.0
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
        
    @setting(4311, 'Display Format', fmt=['s'], returns=['s'])
    def dispFormat(self, c, fmt=None):
        '''Set or get the display format. Following options are allowed:
            "LOGMAG" - log magnitude display
            "LINMAG" - linear magnitude display
            "PHASE" - phase display
            "REIM"  - real and imaginary display
        '''
        dev = self.selectedDevice(c)
        
        if fmt is None:
        
            resp = yield dev.query('LOGM?;')
            if bool(int(resp)):
                fmt = 'LOGMAG'
                returnValue(fmt)
            
            resp = yield dev.query('LINM?;')
            if bool(int(resp)):
                fmt = 'LINMAG'
                returnValue(fmt)
                
            resp = yield dev.query('PHAS?;')
            if bool(int(resp)):
                fmt = 'PHASE'
                returnValue(fmt)
                
            resp = yield dev.query('POLA?;')
            if bool(int(resp)):
                fmt = 'REIM'
                returnValue(fmt)
                
            
        else:
        
            if fmt == 'LOGMAG': 
                yield dev.write('LOGM;')
            elif fmt == 'LINMAG': 
                yield dev.write('LINM;')
            elif fmt == 'PHASE': 
                yield dev.write('PHAS;')
            elif fmt == 'REIM': 
                yield dev.write('POLA;')
            else:
                raise ValueError('Unknown display format request.')
                
        returnValue(fmt)
   
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
        
    @setting(4310, 'Get Trace', returns=['*c[]'])
    def getTrace(self, c):
        '''Get network analyzer trace, output is complex and depends on the display format:
            "LOGMAG" - real: dB, imag: N/A
            "LINMAG" - real: linear units, imag: N/A
            "PHASE" - real: degrees. imag: N/A
            "REIM"  - real: real part (linear) imag: imagninary part (linear)
        '''
        dev = self.selectedDevice(c)
        
        yield dev.write('FORM5;')
        
        avgOn = yield self.averageOnOff(c)
        waitTime = yield dev.query('SWET?;')
        
        if avgOn:
            numAvg = yield self.averagePoints(c)
            sweepWait = numAvg*float(waitTime) + 0.05
            yield self.restartAverage(c)
        else:
            sweepWait = float(waitTime) + 0.05
            
        time.sleep(sweepWait)
        
        yield dev.write('OUTPFORM;')
        dataBuffer = yield dev.read_raw()
        rawData = numpy.fromstring(dataBuffer,dtype=numpy.float32)
        #print rawData
        data = numpy.empty((rawData.shape[-1]-1)/2, dtype=numpy.complex)
                
        data.real = rawData[1:-1:2]
        data.imag = numpy.hstack((rawData[2:-1:2],rawData[-1])) #ugly...
        
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
