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
import numpy

class HP3478AServer(GPIBManagedServer):
    name = 'HP3478A' # Server name
    deviceName = 'Hewlet Packard 3478A' # *IDN? doesnt work for this one, ugh
  
    @setting(11, 'Get Measurement', measurement='s',range='s', trigger='s',returns = '?')
    def getMeasurement(self, c, measurement='DC volts',range='3V,3A', trigger='internal trigger'):
        """Returns the value for the requested measurement.  
        Measurements include {'DC volts', 'AC volts', '2-wire ohms', '4-wire ohms', 'DC current', 'AC current', 'extended ohms'}.
        Range options are {'autorange','30mV','300mV,300mA','3V,3A','30V,30ohms','300V,300ohms','3kohms','30kohms','300kohms','3Mohms','30Mohms'}
        Trigger options are {'internal trigger','single trigger','external trigger','trigger hold','fast trigger'}"""
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
        resp = yield self.HP_query(c,queryString)
        yield self.HP_write(c,'M00') #clear SRQ mode
        if measurement[-5:]=='volts': resp = float(resp)*units.V
        elif measurement[-4:]=='ohms': resp = float(resp)*units.O
        elif measurement[-7:]=='current': resp = float(resp)*units.I
        returnValue(resp)
    
    @inlineCallbacks
    def HP_write(self,c,data):
        """Since *IDN? does not work with this instrument, we cannot select devices or 
        write in the normal way.  This contacts all gpib buses and looks for the selected
        address and writes the data to it."""
        allServers = yield self.client.manager.servers()
        servers = [s for n, s in allServers
                     if (('GPIB Bus' in s) or ('gpib_bus' in s)) ]
        if 'addr' not in c:
            raise DeviceNotSelectedError("No GPIB address selected")
        for serv in servers:
            devices = yield self.client[serv].list_addresses()
            if c['addr'] in devices:
                yield self.client[serv].address(c['addr'])
                yield self.client[serv].write(data)
    
    @inlineCallbacks
    def HP_query(self,c,data):
        allServers = yield self.client.manager.servers()
        servers = [s for n, s in allServers
                     if (('GPIB Bus' in s) or ('gpib_bus' in s)) ]
        if 'addr' not in c:
            raise DeviceNotSelectedError("No GPIB address selected")
        for serv in servers:
            devices = yield self.client[serv].list_addresses()
            if c['addr'] in devices:
                yield self.client[serv].address(c['addr'])
                resp = yield self.client[serv].query(data)
                returnValue( resp )
    
    @setting(12, 'Get Ruox Temperature', returns=['v[K]'])
    def getRuoxTemperature(self, c):
        """Get the temperatures of the Ruox Thermometer for the ADR fridge.  All RuOx readers of every kind must have this method to work with the ADR control program."""
        reg = self.client.registry
        reg.cd(c['adr settings path'])
        RCal = yield reg.get('RCal')
        V = yield self.getMeasurement(c,'DC volts','3V,3A')
        R = RCal*1000*V['V']
        try: T = pow((2.85/math.log((R-652)/100)),4)*units.K
        except ValueError: T = numpy.nan*units.K
        returnValue(T)
    
    @setting(2,'Select Device',addr=['s'], returns=['s'])
    def select_device(self, c, addr=None):
        """Get or set the GPIB address for this context.

        Note: The list devices function will not work for this device.
        """
        if addr is not None:
            c['addr'] = addr
        return c['addr']
    
    @setting(15,'Set ADR Settings Path',path=['*s'])
    def set_adr_settings_path(self,c,path):
        c['adr settings path'] = path
  
__server__ = HP3478AServer()
  
if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)