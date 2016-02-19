# Copyright (C) 2011 Dylan Gorman
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
name = Keithley 2000 DMM
version = 1.0
description = 
  
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
import numpy
import math
  
class KeithleyWrapper(GPIBDeviceWrapper):
    @inlineCallbacks
    def initialize(self):
        self.dcVolts = yield self.getdcVolts()
        self.outputStateKnown = False
        self.output = True
  
    @inlineCallbacks
    def getdcVolts(self):
        resp = yield self.query('MEAS:VOLT:DC?')
        self.dcVolts = float(resp.split(',')[0].strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        returnValue(self.dcVolts*units.V)
        
    @inlineCallbacks
    def getRes(self):
        resp = yield self.query('MEAS:RES?')
        self.ohms = float(resp.split(',')[0].strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        returnValue(self.ohms*units.Ohm)
        
    @inlineCallbacks
    def getFourWireRes(self):
        resp = yield self.query('MEAS:FRES?')
        self.ohms = float(resp.split(',')[0].strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        returnValue(self.ohms*units.Ohm)
  
class KeithleyServer(GPIBManagedServer):
    name = 'Keithley 2000 DMM' # Server name
    deviceName = ['KEITHLEY INSTRUMENTS INC. MODEL 2000', 'KEITHLEY INSTRUMENTS INC. MODEL 2100']
    deviceWrapper = KeithleyWrapper
  
    @setting(10, 'DC Voltage')
    def dcVolts(self, c):
        """Returns voltage last recorded, but does not acquire again."""
        dev = self.selectedDevice(c)
        return dev.dcVolts
  
    @setting(11, 'Get DC Volts', returns = 'v')
    def getdcVolts(self, c):
        """Aquires new value for DC Voltage and returns it."""
        dev = self.selectedDevice(c)
        voltage = yield dev.getdcVolts()
        returnValue(voltage)
  
    @setting(12, 'Get Resistance', returns = 'v')
    def getResistance(self, c):
        """Aquires resistance and returns it."""
        dev = self.selectedDevice(c)
        res = yield dev.getRes()
        returnValue(res)
  
    @setting(13, 'Get FW Resistance', returns = 'v')
    def getResistance(self, c):
        """Aquires resistance using four=wire measurement and returns it."""
        dev = self.selectedDevice(c)
        res = yield dev.getFourWireRes()
        returnValue(res)
        
    @setting(20, 'Get Ruox Temperature', returns=['v[K]'])
    def getRuoxTemperature(self, c):
        """Get the temperatures of the Ruox Thermometer for the ADR fridge.  All RuOx readers of every kind must have this method to work with the ADR control program."""
        reg = self.client.registry
        reg.cd(c['adr settings path'])
        RCal = yield reg.get('RCal')
        dev = self.selectedDevice(c)
        V = yield dev.getdcVolts()
        R = RCal*1000*V['V']
        try: T = pow((2.85/math.log((R-652)/100)),4)*units.K
        except ValueError: T = numpy.nan*units.K
        returnValue(T)
    
    @setting(21,'Set ADR Settings Path',path=['*s'])
    def set_adr_settings_path(self,c,path):
        c['adr settings path'] = path
  
__server__ = KeithleyServer()
  
if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)