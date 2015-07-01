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
name = Agilent 5230A Network Analyzer.
version = 0.10
description = Four channel 5230A PNA-L network analyzer server

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

class Agilent5230AServer(GPIBManagedServer):
    name = 'Agilent 5230A Network Analyzer'
    deviceName = 'AGILENT 5230A'
    deviceWrapper = GPIBDeviceWrapper
    
    defaultTimeout = 0.05
    
    @setting(600, 'Preset')
    def preset(self, c):
    	"""Performs preset on network analyzer."""
    	dev = self.selectedDevice(c)
    	yield dev.write('SYSTem:PRESet')
    	sleep(defaultTimeout)
    
    @setting(601, 'Power Output', pow=['b'], returns=['b'])
    def power_output(self, c, pow=None):
    	"""Turn output power on or off, or query state."""
    	dev = self.selectedDevice(c)
    	if pow is None:
    		resp = yield dev.query('OUTP?')
    		pow = bool(int(resp))
    	else:
    		if pow:
    			yield dev.write('OUTP ON')
    		else:
    			yield dev.write('OUTP OFF')
    	returnValue(pow)
    	
	@setting(602, 'Center Frequency', cfreq=['v[Hz]'], returns=['v[Hz]'])
	def center_frequency(self, c, cfreq=None):
		"""Set or get the sweep center frequency."""
		dev = self.selectedDevice(c)
		if cfreq is None:
			resp = yield dev.query('SENSe1:FREQuency:CENTer?')
			cfreq = float(resp) * units.Hz
		else:
			yield dev.write('SENSe1:FREQuency:CENTer %i'%cfreq['Hz'])
		returnValue(cfreq)
	
	@setting(603, 'Frequency Span', span=['v[Hz]'], returns=['v[Hz]'])
	def frequency_span(self, c, span=None):
		"""Set or get the sweep center frequency."""
		dev = self.selectedDevice(c)
		if span is None:
			resp = yield dev.query('SENSe1:FREQuency:SPAN')
			span = float(resp) * units.Hz
		else:
			yield dev.write('SENSe1:FREQuency:SPAN %i'%span['Hz'])
		returnValue(cfreq)
    	
    @setting(604, 'Start Frequency', start=['v[Hz]'], returns=['v[Hz]'])
    def start_frequency(self, c, start=None):
    	"""Set or get sweep start frequency."""
    	dev = self.selectedDevice(c)
    	if start is None:
    		resp = yield dev.query('SENSe1:FREQuency:STARt?')
    		start = float(resp) * units.Hz
    	else:
    		yield dev.write('SENSe1:FREQuency:STARt %i'%start['Hz'])
    	returnValue(start)
    	
    @setting(605, 'Stop Frequency', stop=['v[Hz]'], returns=['v[Hz]'])
    def stop_frequency(self, c, stop=None):
    	"""Set or get sweep stop frequency."""
    	dev = self.selectedDevice(c)
    	if start is None:
    		resp = yield dev.query('SENSe1:FREQuency:STOP')
    		stop = float(resp) * units.Hz
    	else:
    		yield dev.write('SENSe1:FREQuency:STOP %i'%stop['Hz'])
    	returnValue(start)
    	
    @setting(606, 'Sweep Type', stype=['s'], returns['s'])
    def sweep_type(self, c, stype=None):
    	"""Set or get the frequency sweep type. 'LIN' - for linear, 'CW' - for single frequency."""
    	dev = self.selectedDevice(c)
    	if stype is None:
    		stype = yield dev.query('SENSe1:SWEep:TYPE?')
    	else:
    		if (stype.upper() ~= 'CW') and (stype.upper() ~= 'LIN'):
    			raise ValueError('Unknown sweep type: ' + str(stype) + '. Please use "LIN" or "CW".')
    		else:
    			yield dev.write('SENSe1:SWEep:TYPE ' + stype)
    	returnValue(type)
    
    @setting(607, 'IF Bandwidth', bw=['v[Hz]'], returns=['v[Hz]'])
    def if_bandwidth(self, c, bw=None):
    	"""Set or get the IF bandwidth."""
    	dev = self.selectedDevice(c)
    	if bw is None:
    		resp = yield dev.query('SENSe1:BANDwidth?')
    		bw = float(resp) * units.Hz
    	else:
    		yield dev.write('SENSe1:BANDwidth %i'%bw['Hz'])
    	returnValue(type)
    
    @setting(608, 'Average Mode', avg=['b'], returns=['b'])
    def average_mode(self, c, avg=None):
    	"""Turn sweep averaging on or off, or query state."""
    	dev = self.selectedDevice(c)
    	if avg is None:
    		resp = yield dev.query('SENSe1:AVERage?')
    		avg = bool(int(resp))
    	else:
    		if avg:
    			yield dev.write('SENSe1:AVERage ON')
    		else:
    			yield dev.write('SENSe1:AVERage OFF')
    	returnValue(avg)
    	
    @setting(609, 'Restart Averaging')
    def restart_averaging(self, c):
    	"""Clears and restarts trace averaging on the current sweep."""
    	dev = self.selectedDevice(c)
    	yield dev.write('SENSe1:AVERage:CLEar')
    
    @setting(610 'Average Points', count=['w'], returns=['i'])
    def average_points(self, c, count=None):
    	"""Set or get the number of measurements to combine for an average."""
    	dev = self.selectedDevice(c)
    	if count is None:
    		resp = yield dev.query('SENSe1:AVER:COUN?')
    		count = int(float(resp))
    	else:
    		yield dev.write('SENSe1:AVER:COUN %d'%count)
    	returnValue(count)
    
    @setting(611, 'Source Power', pow=['v[dBm]'], returns=['v[dBm]'])
    def source_power(self, c, pow=None):
    	"""Set or get source RF power."""
    	dev = self.selectedDevice(c)
    	if pow is None:
    		resp = yield dev.query('SOURce:POWer?')
    		pow = float(resp) * units.dBm
    	else:
    		yield dev.write('SOURce:POW1 %f'%pow['dBm'])
    	returnValue(pow)
    	
    @setting(612, 'Get Sweep Time', returns=['v[s]'])
    def get_sweep_time(self, c):
    	"""Get the time to complete a sweep."""
    	dev = self.selectedDevice(c)
       	resp = yield dev.query('SENSe1:SWEep:TIME?')
    	swpTime = float(resp) * units.s
    	returnValue(swpTime)
    
    @setting(613, 'Sweep Points', points=['i'], returns=['i'])
    def sweep_points(self, c, points=None):
    	"""Set or get the number of points in the sweep."""
    	dev = self.selectedDevice(c)
    	if points is None:
    		resp = yield dev.query('SENSe1:SWEep:POINts?')
    		points = int(float(resp))
    	else:
    		yield dev.write('SENSe1:SWEep:POINts %i'% points)
    	returnValue(points)
    		    	
    @setting(614, 'Measurement Setup', meas=['s'])
    def measurement_setup(self, c, meas='S21'):
    	"""Set the measurement parameters. Use a string of the form Sxx (S21, S11...) for
    		the measurement type.
    	"""
    	if meas not in ('S11', 'S12', 'S13', 'S14', 'S21', 'S22', 'S23', 'S24', 
    					 'S31', 'S32', 'S33', 'S34', 'S41', 'S42', 'S43', 'S44'):
    		raise ValueError('Illegal measurment definition: %s'% str(meas))

        dev = self.selectedDevice(c)            
    	yield dev.write('CALC:PAR:DEL:ALL')
    	yield dev.write('DISPlay:WINDow1:STATE ON')
    	yield dev.write('CALCulate:PARameter:DEFine:EXT "Meas" ,%s'% meas)
    	yield dev.write('DISPlay:WINDow1:TRACe1:FEED "MyMeas"')
    	yield dev.write('CALC:PAR:SEL "MyMeas"')
    
    @setting(615, 'Get Trace', returns=['*v[]'])
    def get_trace(self,c):
    	"""Get the active trace from the network analyzer."""
    	dev = self.selectedDevice(c)    	
    	meas = yield dev.query('SYST:ACT:MEAS?')
    	yield dev.write('CALC:PAR:SEL %s'% meas)   
    	yield dev.write('FORM REAL,32')	
    	
    	avgOnOff = yield self.average_mode()
    	swpTime  = yield self.get_sweep_time()
    	
    	if avgOnOff:
    		avgCount = yield self.average_points()
    		yield self.restart_averaging()
    		sleepTime = avgCount * swpTime + defaultTimeout
    		sleep(sleepTime)
    	else:
    		sleep(swpTime + defaultTimeout)

    	yield dev.write('CALC1:DATA? FDATA')
    	rawDataBlock = yield dev.read_raw()
    	
    	rawData = numpy.fromstring(rawDataBlock, dtype=numpy.float32)
    	data = rawData[3:-1]
    	
    @setting(599, 'Initialize')
    def initialize(self, c):
        """Initialize the network analyzer."""
        dev = self.selectedDevice(c)
        yield self.preset(c)


__server__ = Agilent5230AServer()


if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)