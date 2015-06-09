# Copyright (C) 2015 Chris Wilen
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>
  
"""
### BEGIN NODE INFO
[info]
name = HP3478A
version = 1.0
description = Provides a way to control and get data from the HP3478A Temperature Monitor.
  
[startup]
cmdline = %PYTHON% %FILE%
timeout = 20
  
[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""
  
from labrad.server import setting
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from twisted.internet.defer import inlineCallbacks, returnValue
from labrad import units
import math


class HP3478AWrapper(GPIBDeviceWrapper):
  
    @inlineCallbacks
    def getMeasurement(self, measurement='DC volts',range='3V,3A', trigger='internal trigger'):
        measurements = {'DC volts':0,'AC volts':1,'2-wire ohms':2,'4-wire ohms':3,'DC current':4,'AC current':5,'extended ohms':6}
        ranges = {'autorange':'A','30mV':-2,'300mV,300mA':-1,'3V,3A':0,'30V,30ohms':1,'300V,300ohms':2,'3kohms':3,'30kohms':4,'300kohms':5,'3Mohms':6,'30Mohms':7}
        triggers = {'internal trigger':0,'single trigger':1,'external trigger':2,'trigger hold':3,'fast trigger':4}
        if measurement not in measurements:
            raise Exception('Not a valid measurement!')
        if range not in ranges:
            raise Exception('Not a valid range!')
        if trigger not in triggers:
            raise Exception('Not a valid trigger setting!')
        # SRQ on data available(M01), autozero(Z1),5 1/2 digit display,
        # greatest noise rejection and 10 PLC integeration(N5)
        queryString = 'M01Z1N5F%iR%sT%i'%(measurements[measurement],ranges[range],triggers[trigger])
        resp = yield self.query(queryString)
        yield self.write('M00') #clear SRQ mode
        if measurement[-5:]=='volts': resp = float(resp)*units.V
        elif measurement[-4:]=='ohms': resp = float(resp)*units.O
        elif measurement[-7:]=='current': resp = float(resp)*units.I
        returnValue(resp)
  
class HP3478AServer(GPIBManagedServer):
    name = 'HP3478A' # Server name
    deviceName = 'Hewlet Packard 3478A' # Model string returned from *IDN?
    deviceWrapper = HP3478AWrapper
  
    @setting(11, 'Get Measurement', measurement='s',range='s', trigger='s',returns = '?')
    def getMeasurement(self, c, measurement='DC volts',range='3V,3A', trigger='internal trigger'):
        """Returns the value for the requested measurement.  
        Measurements include {'DC volts', 'AC volts', '2-wire ohms', '4-wire ohms', 'DC current', 'AC current', 'extended ohms'}.
        Range options are {'autorange','30mV','300mV,300mA','3V,3A','30V,30ohms','300V,300ohms','3kohms','30kohms','300kohms','3Mohms','30Mohms'}
        Trigger options are {'internal trigger','single trigger','external trigger','trigger hold','fast trigger'}"""
        dev = self.selectedDevice(c)
        resp = yield dev.getMeasurement(measurement,range,trigger)
        returnValue(resp)
    
    @setting(12, 'Get Ruox Temperature', returns=['*v[K]'])
    def getRuoxTemperature(self, c):
        """Get the temperatures of the Ruox Thermometer for the ADR fridge.  All RuOx readers of every kind must have this method to work with the ADR control program."""
        reg = self.client.registry
        reg.cd(['','ADR Settings','ADR Square'])
        RCal = yield reg.get('RCal')
        dev = self.selectedDevice(c)
        V = yield dev.getMeasurement('DC volts','3V,3A')
        R = RCal*1000*V
        T = pow((2.85/math.log((T-652)/100)),4)*units.K
        returnValue(T)
  
__server__ = HP3478AServer()
  
if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)