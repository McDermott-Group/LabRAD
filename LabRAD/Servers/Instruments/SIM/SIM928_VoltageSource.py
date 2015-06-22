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
#
# TODO: if we want to be able to use multiple of these (when we get another
# fridge, for example), we will need to add context support to remember the channel, lastTime.

"""
### BEGIN NODE INFO
[info]
name = SIM928 Server
version = 2.1
description = This serves as an interface for the SIM928 Voltage Source.

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
import labrad.units as units

class SIM928Server(GPIBManagedServer):
    """Provides basic control for SRS 925 Multiplexer Module"""
    name = 'SIM928 Server'
    deviceName = 'Stanford_Research_Systems SIM928' #Stanford_Research_Systems,SIM928,s/n105794,ver3.6\n
    deviceWrapper = GPIBDeviceWrapper
    
    @setting(101, 'Voltage', v=['v[V]'], returns=['v[V]'])
    def voltage(self, c, v=None):
        """Get or set the voltage."""
        dev = self.selectedDevice(c)
        if v is None:
            resp = yield dev.query('VOLT?')
            v = float(resp)*units.V
        else:
            yield dev.write('VOLT %f' % v['V'])
        returnValue(v)
        
    @setting(102, 'Output', on=['b'], returns=['b'])
    def output(self, c, on=None):
        """Get or set the output (on/off)."""
        dev = self.selectedDevice(c)
        if on is None:
            resp = yield dev.query('EXON?')
            on = bool(resp)
        else:
            yield dev.write('EXON %i' % int(on))
        returnValue(on)

__server__ = SIM928Server()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
