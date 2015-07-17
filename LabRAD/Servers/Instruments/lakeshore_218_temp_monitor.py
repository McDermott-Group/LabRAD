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
name = Lakeshore 218
version = 1.0
description = Provides a way to control and get data from the Lakeshore 218 Temperature Monitor.
  
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

class Lakeshore218Wrapper(GPIBDeviceWrapper):
  
    @inlineCallbacks
    def getTemp(self, channel=0, unit='K'):
        # command: type#
        # type = K:'KRDG?', C: 'CRDG?', sensor units: 'SRDG?', linear data:'LRDG?'
        # # = channel from which to read, 0 for all
        unit = unit.upper()
        if unit not in ['K','C','S','L']:
            raise Exception('Not a valid unit!')
        resp = yield self.query(unit+'RDG?%i'%channel)
        if unit=='K': temp = [float(t)*units.K for t in resp.split(',')]
        elif unit=='C': temp = [float(t)*units.C for t in resp.split(',')]
        elif unit=='S': temp = [float(t)*units.V for t in resp.split(',')]
        else: temp = resp
        returnValue(temp)
  
class Lakeshore218Server(GPIBManagedServer):
    name = 'Lakeshore 218' # Server name
    deviceName = 'LSCI MODEL218S' # Model string returned from *IDN?
    deviceWrapper = Lakeshore218Wrapper
  
    @setting(11, 'Get Temperature', channel='v',unit='v',returns = '?')
    def getTemp(self, c, channel=0,unit='K'):
        """Returns the temperatures for a given channel (or a list of all channels if channel=0) in the specified units (K,C,Linear(L),System(S))."""
        dev = self.selectedDevice(c)
        temp = yield dev.getTemp(channel,unit)
        returnValue(temp)
    
    @setting(12, 'Get Diode Temperatures', returns=['*v[K]'])
    def getDiodeTemperatures(self, c):
        """Get the temperatures of the Si Diode Thermometers."""
        reg = self.client.registry
        reg.cd(c['adr settings path'])
        channels = yield reg.get('Lakeshore Diode Channels')
        dev = self.selectedDevice(c)
        tempArray = yield dev.getTemp()
        returnValue( [tempArray[c-1] for c in channels] )
    
    @setting(13, 'Get Magnet Voltage', returns=['v[V]'])
    def getMagnetVoltage(self, c):
        """Get the voltage across the ADR magnet."""
        reg = self.client.registry
        reg.cd(c['adr settings path'])
        channels = yield reg.get('Lakeshore Magnet Voltage Channels')
        dev = self.selectedDevice(c)
        magVArray = yield dev.getTemp(unit='S')
        voltages = [magVArray[c-1] for c in channels]
        maxV = max(voltages)
        if voltages[0] < voltages[1]: maxV = -maxV
        returnValue( abs(maxV)*units.V )
        
    @setting(15,'Set ADR Settings Path',path=['*s'])
    def set_adr_settings_path(self,c,path):
        c['adr settings path'] = path
  
__server__ = Lakeshore218Server()
  
if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)