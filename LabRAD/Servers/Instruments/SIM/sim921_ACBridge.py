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
name = SIM921 Server
version = 2.2
description = This SIM921 AC Resistance Bridge is used to measure the RuOx Thermometers.

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

class SIM921Server(GPIBManagedServer):
    """Provides basic control for SRS SIM921 AC Resistance Bridge Module"""
    name = '%LABRADNODE% SIM921'
    deviceName = 'STANFORD RESEARCH SYSTEMS SIM921' # *IDN? = "Stanford_Research_Systems,SIM921,s/n105794,ver3.6"
    deviceWrapper = GPIBDeviceWrapper

    @setting(101, 'Get Time Constant', returns=['v[mS]'])
    def getTimeConstant(self, c):
        """Get the time constant (in ms) currently set for the AC Res Bridge."""
        dev = self.selectedDevice(c)
        timeConstCodes = {-1:'filter off', 0:0.3, 1:1, 2:3, 3:10, 4:30, 5:100, 6:300}
        returnCode = yield dev.query("TCON?")
        t = timeConstCodes[int(returnCode)]
        returnValue(t)

    @setting(102, 'Get Ruox Temperature', returns=['v[K]'])
    def getRuoxTemperature(self, c):
        """Get temperature being read by the AC Res Bridge right now."""
        dev = self.selectedDevice(c)
        gpibstring = yield dev.query("TVAL?")
        T = float(gpibstring.strip('\x00'))
        returnValue( T )
    
    @setting(103, 'Set Curve', curve=['v'])
    def setCurve(self, c, curve):
        """Get temperature being read by the AC Res Bridge right now."""
        dev = self.selectedDevice(c)
        yield dev.write("CURV %d" %channel)

__server__ = SIM921Server()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
