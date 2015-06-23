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
name = SIM925 Server
version = 2.2
description = This serves as an interface for the SIM921 AC Resistance Bridge used to measure the RuOx Thermometers.

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
import time

class SIM925Server(GPIBManagedServer):
    """Provides basic control for SRS SIM925 Multiplexer Module"""
    name = 'SIM925'
    deviceName = 'STANFORD RESEARCH SYSTEMS SIM925' # *IDN? = "Stanford_Research_Systems,SIM921,s/n105794,ver3.6"
    deviceWrapper = GPIBDeviceWrapper
    
    def __init__(self):
        GPIBManagedServer.__init__(self)
        self.channel = 0 #channel 2 is the FAA pill.  GGG pill is chan 1
        self.lastTime = time.time()
        #self.channel(2) &&& fix this

    @setting(101, 'Get Time Since Channel Set', returns=['v[ms]'])
    def getTimeSinceChannelSet(self, c):
        """Get the time since the current channel was last changed."""
        return time.time() - self.lastTime

    @setting(102, 'Channel', channel=['v'], returns=['v'])
    def channel(self, c, channel=None):
        """Get or set the current channel."""
        if channel != None and channel != self.channel:
            self.channel = channel
            dev = self.selectedDevice(c)
            yield dev.write("CHAN %d" %channel)
            self.lastTime = time.time()
        returnValue( self.channel )

__server__ = SIM925Server()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
