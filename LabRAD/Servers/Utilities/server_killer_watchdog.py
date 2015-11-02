# Copyright (C) 2015 Ivan Pechenezhskiy
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
name = Server Killer Watchdog
version = 0.1.0
description =  Exterminates slow or stupid beasts.
instancename = Server Killer Watchdog

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

import os
import time
import numpy as np

from twisted.internet.defer import inlineCallbacks
from twisted.internet.reactor import callLater

from labrad.server import LabradServer, setting
from labrad.units import s


class ServerKillerWatchdog(LabradServer):
    """
    This server exterminates slow and/or stupid beasts with
    the LabRAD node.
    """
    name = 'Server Killer Watchdog'
    
    @inlineCallbacks
    def getRegistryKeys(self):
        """
        Get registry keys for the Leiden DR Temperature Pseudoserver.
        """
        reg = self.client.registry()
        comp_name = os.environ['COMPUTERNAME'].lower()
        yield reg.cd(['', 'Servers', 'Server Killer Watchdog',
                comp_name], True)
        dirs, keys = yield reg.dir()
        if 'LabRAD node' in keys:
            self._node = yield reg.get('LabRAD node')
        else:
            self._node = 'node ' + comp_name

        try:
            yield self.client[self._node].status()
        except:
            raise Exception("Could not communicate with the LabRAD " +
                    "node server '" + str(self._node) + "'")
        print("LabRAD node name is set to " + str(self._node))

    @inlineCallbacks    
    def initServer(self):
        """Initialize the Server Killer Watchdog Pseudoserver."""
        yield self.getRegistryKeys()
        self._timer_start_time = {}

    @inlineCallbacks
    def killServer(self, server):
        if server in self._timer_start_time:
            self._timer_start_time.pop(server)
            servs = yield self.client[self._node].running_servers()
            if server in [serv for pairs in servs for serv in pairs]:
                yield self.client[self._node].stop(server)
                print("'" + server + "' has been stopped.")

    @setting(10, 'Start Timer', server='s', timeout='v[s]')
    def start_timer(self, c, server, timeout=60*s):
        """Set a timeout timer for a server."""
        self._timer_start_time[server] = time.time()
        callLater(timeout['s'], self.killServer, server)

    @setting(11, 'Stop Timer', server='s', returns='v[s]')
    def stop_timer(self, c, server):
        """Stop the timeout timer and return the elapsed time."""
        if server in self._timer_start_time:
            elapsed_time = time.time() - self._timer_start_time[server]
            self._timer_start_time.pop(server)
        else:
            elapsed_time = np.nan
        return elapsed_time * s


__server__ = ServerKillerWatchdog()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)