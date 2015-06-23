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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
### BEGIN NODE INFO
[info]
name = SIM922 Server
version = 2.2
description = This Diode Temperature Monitor is used to measure the Si Diode thermometers in the ADR, as well as the voltage across the magnet.

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

class SIM922Server(GPIBManagedServer):
    """Provides basic control for SRS SIM922 Diode Temperature Monitor Module"""
    name = 'SIM922'
    deviceName = 'STANFORD RESEARCH SYSTEMS SIM922' # *IDN? = "Stanford_Research_Systems,SIM922,s/n105794,ver3.6"
    deviceWrapper = GPIBDeviceWrapper

    @setting(101, 'Get Diode Temperatures', returns=['*v[K]'])
    def getDiodeTemperatures(self, c):
        """Get the temperatures of the Si Diode Thermometers connected to the first two slots of the SIM922."""
        dev = self.selectedDevice(c)
        diodeMonitorReturnString = yield dev.query("TVAL? 0")
        temperatures = [float(x) for x in diodeMonitorReturnString.strip('\x00').split(',')][:2]
        returnValue( temperatures )

    @setting(102, 'Get Magnet Voltage', returns=['v[V]'])
    def getMagnetVoltage(self, c):
        """Get the voltage across the magnet.  Two values are measured (third and fourth slots in the SIM922)
           and averaged for the returned result."""
        dev = self.selectedDevice(c)
        dev.write("*CLS")
        diodeMonitorReturnString = yield dev.query("VOLT? 0")
        magnetVoltages = [float(x) for x in diodeMonitorReturnString.strip('\x00').split(',')][2:]
        returnValue( (abs(magnetVoltages[0])+abs(magnetVoltages[1]))/2 )

__server__ = SIM922Server()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
