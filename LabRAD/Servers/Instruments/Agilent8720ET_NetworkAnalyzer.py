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
name = Agilent 8720ET Network Analyzer
version = 1.1
description = Two channel 8720ET transmission/reflection network analyzer server.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import os.path
if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    SCRIPT_PATH = os.path.dirname(os.getcwd())  # This will be executed when the script is loaded by the labradnode.
else:
    SCRIPT_PATH = os.path.dirname(__file__)     # This will be executed if the script is started by clicking or in a command line.
NONBLOCKING_PATH = os.path.join(SCRIPT_PATH.rsplit('LabRAD', 1)[0], 'LabRAD\Servers\Utilities')
import sys
if NONBLOCKING_PATH not in sys.path:
    sys.path.append(NONBLOCKING_PATH)

import numpy

from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from labrad.server import setting, returnValue
import labrad.units as units

from nonblocking import sleep

class Agilent8720ETServer(GPIBManagedServer):
    name = 'Agilent 8720ET Network Analyzer'
    deviceName = 'HEWLETT PACKARD 8720ET'
    deviceWrapper = GPIBDeviceWrapper
    
    @setting(431, 'Preset')
    def preset(self, c):
        """Reset the network analyzer."""
        dev = self.selectedDevice(c)
        yield dev.query('OPC?;PRES')
        
    @setting(432, 'Start Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def start_frequency(self, c, freq=None):
        """Set or get the start frequency of the sweep."""
        dev = self.selectedDevice(c)
        if freq is None:
            resp = yield dev.query('STAR?')
            freq = float(resp) * units.Hz
        else:
            yield dev.write('STAR%i' %freq['Hz'])
        returnValue(freq)
    
    @setting(433, 'Stop Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def stop_frequency(self, c, freq=None):
        """Set or get the stop frequency of the sweep."""
        dev = self.selectedDevice(c)
        if freq is None:
            resp = yield dev.query('STOP?')
            freq = float(resp) * units.Hz
        else:
            yield dev.write('STOP%i' %freq['Hz'])
        returnValue(freq)
        
    @setting(435, 'Measurement Setup', mode=['s'], returns=['s'])
    def measurement_setup(self, c, mode=None):
        """Set or get the measurement mode: transmission or reflection. 
            Following options are allowed (could be in any letter case):
            "REFL", 'R', 'REFLECTION' - reflection mode;
            "TRAN", 'T', 'TRANSMISSION', 'TRANS' - transmission mode.
        
            Output is normally either 'TRANSMISSION' or 'REFLECTION'.
        """
        dev = self.selectedDevice(c)
        if mode is None:
            resp = yield dev.query('RFLP?')
            if bool(int(resp)):
                mode = 'REFLECTION'
            else:
                mode = 'TRANSMISSION'
        else:
            if mode.upper() in ['R', 'REFL', 'REFLECTION']:
                mode = 'REFLECTION'
                yield dev.write('RFLP')
            if mode.upper() in ['T', 'TRAN', 'TRANS', 'TRANSMISSION']:
                yield dev.write('TRAP')
                mode = 'TRANSMISSION'
            else:
                raise ValueError('Unknown measurement mode: ' + str(mode))
        returnValue(mode)

    @setting(436, 'Sweep Points', pn=['w'], returns=['w'])
    def sweep_points(self, c, pn=None):
        """Set or get number of points in a sweep. The number will be automatically 
        coarsen to 3, 11, 21, 26, 51, 101, 201, 401, 801, or 1601 by the network analyzer."""
        dev = self.selectedDevice(c)
        if pn is None:
            resp = yield dev.query('POIN?')
            pn = int(float(resp))
        else:
            yield dev.write('POIN%i' %pn)
            t = yield dev.query('SWET?')
            sleep(2 * float(t)) # Be sure to wait for two sweep times as required in 8270ET docs.
        returnValue(pn)   
    
    @setting(437, 'Average Mode', avg=['b'], returns=['b'])
    def average_mode(self, c, avg=None):
        """Set or get average state."""
        dev = self.selectedDevice(c)
        if avg is None:
            resp = yield dev.query('AVERO?')
            avg = bool(int(resp))
        else:
            if avg:
                yield dev.write('AVEROON')
            else:
                yield dev.write('AVEROOFF')
        returnValue(avg)
        
    @setting(438, 'Average Points', aN=['w'], returns=['w'])
    def average_points(self, c, aN=None):
        """Set or get number of points in average (in 0-999 range)."""
        dev = self.selectedDevice(c)
        if aN is None:
            N = yield dev.query('AVERFACT?')
            aN = int(float(N))
        else:
            yield dev.write('AVERFACT%i' %aN)
        returnValue(aN)
        
    @setting(439, 'Restart Averaging')
    def restart_averaging(self, c):
        """Restart trace averaging."""
        dev = self.selectedDevice(c)
        yield dev.write('AVERREST')

    @setting(445, 'Power', pow=['v[dBm]'], returns=['v[dBm]'])
    def power(self, c, pow=None):
        """Set or get the sweep power level."""
        dev = self.selectedDevice(c)
        if pow is None:
            resp = yield dev.query('POWE?')
            pow = float(resp) * units.dBm
        else:
            if pow['dBm'] < -10:
                print('Minimum power level for Agilent 8720ET is -10 dBm.')
                pow = -10 * units.dBm;
            if pow['dBm'] > 10:
                print('Maximum power level for Agilent 8720ER is 10 dBm.')
                pow = 10 * units.dBm;
            yield dev.write('POWE%iDB' %pow['dBm'])
        returnValue(pow)
        
    @setting(450, 'Get Trace', returns=['*c[]'])
    def get_trace(self, c):
        """Get network analyzer trace. The output is complex and depends on the display format:
            "LOGMAG" - real: dB, imag: N/A;
            "LINMAG" - real: linear units, imag: N/A;
            "PHASE"  - real: degrees, imag: N/A;
            "REIM"   - real: real part (linear), imag: imaginary part (linear).
        """
        dev = self.selectedDevice(c)
        yield dev.write('FORM5')
        avgOn = yield self.average_mode(c)
        waitTime = yield dev.query('SWET?')
        if avgOn:
            numAvg = yield self.average_points(c)
            sweepWait = numAvg * float(waitTime) + 0.05
            yield self.restart_averaging(c)
        else:
            sweepWait = float(waitTime) + 0.05 
        sleep(sweepWait)
        yield dev.write('OUTPFORM')
        dataBuffer = yield dev.read_raw()
        rawData = numpy.fromstring(dataBuffer,dtype=numpy.float32)
        data = numpy.empty((rawData.shape[-1] - 1) / 2, dtype=numpy.complex)
        data.real = rawData[1:-1:2]
        data.imag = numpy.hstack((rawData[2:-1:2], rawData[-1])) #ugly...
        returnValue(data)
        
    @setting(451, 'Display Format', fmt=['s'], returns=['s'])
    def display_format(self, c, fmt=None):
        """Set or get the display format. Following options are allowed:
            "LOGMAG" - log magnitude display;
            "LINMAG" - linear magnitude display;
            "PHASE"  - phase display;
            "REIM"   - real and imaginary display.
        """
        dev = self.selectedDevice(c)
        if fmt is None:
            resp = yield dev.query('LOGM?')
            if bool(int(resp)):
                fmt = 'LOGMAG'
                returnValue(fmt)
            resp = yield dev.query('LINM?')
            if bool(int(resp)):
                fmt = 'LINMAG'
                returnValue(fmt)
            resp = yield dev.query('PHAS?')
            if bool(int(resp)):
                fmt = 'PHASE'
                returnValue(fmt)
            resp = yield dev.query('POLA?')
            if bool(int(resp)):
                fmt = 'REIM'
                returnValue(fmt)
        else:
            if fmt.upper() == 'LOGMAG': 
                yield dev.write('LOGM')
            elif fmt.upper() == 'LINMAG': 
                yield dev.write('LINM')
            elif fmt.upper() == 'PHASE': 
                yield dev.write('PHAS')
            elif fmt.upper() == 'REIM': 
                yield dev.write('POLA')
            else:
                raise ValueError('Unknown display format request: ' + str(fmt))
        returnValue(fmt)

    @setting(99, 'Initialize')
    def initialize(self, c):
        """Initialize the network analyzer."""
        dev = self.selectedDevice(c)
        yield self.preset(c)
        yield dev.write('CHAN1;AUXCOFF;S21;LOGM;AUTO')
        yield self.sweep_points(c, 801)
        print('Initialized...')


__server__ = Agilent8720ETServer()


if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)