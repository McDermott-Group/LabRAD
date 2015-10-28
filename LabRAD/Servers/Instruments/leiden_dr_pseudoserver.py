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
name = Leiden DR Temperature
version = 0.2.1
description =  Gives access to Leiden DR temperatures.
instancename = Leiden DR Temperature

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

import os

from twisted.internet.defer import inlineCallbacks
from twisted.internet.reactor import callLater
from twisted.internet.task import LoopingCall

from labrad.server import LabradServer, setting
from labrad.units import mK, s


class LeidenDRPseudoserver(LabradServer):
    """
    Provides access to the Leiden DR temperatures by reading log
    files on the AFS.
    """
    name = 'Leiden DR Temperature'
    refreshInterval = 15 * s
    
    @inlineCallbacks
    def getRegistryKeys(self):
        """
        Get registry keys for the Leiden DR Temperature Pseudoserver.
        """
        reg = self.client.registry()
        yield reg.cd(['', 'Servers', 'Leiden DR Temperature'], True)
        dirs, keys = yield reg.dir()
        if 'Leiden Log Files Path' in keys:
            self._path = yield reg.get('Leiden Log Files Path')
            
        if ('Leiden Log Files Path' not in keys or 
                not os.path.exists(self._path)):
                self._path = ('\\AFS\physics.wisc.edu\mcdermott-group' + 
                              '\Data\DR Log Files\Leiden')
                
        if not os.path.exists(self._path):
            raise Exception("Could not find the Leiden Log Files Path: '" +
                    str(self._path) + "'")
        print("Leiden Log Files Path is set to " + str(self._path))

    @inlineCallbacks    
    def initServer(self):
        """Initialize the Leiden DR Temperature Pseudoserver."""
        yield self.getRegistryKeys()
        self._offset = 1024 # Number of bytes to read near the end 
                            # of the log file.
        yield self.readTemperatures()
        callLater(0.1, self.startRefreshing)

    def startRefreshing(self):
        """
        Start periodically refreshing the temperatures.

        The start call returns a deferred which we save for later.
        When the refresh loop is shutdown, we will wait for this
        deferred to fire to indicate that it has terminated.
        """
        self.refresher = LoopingCall(self.readTemperatures)
        self.refresherDone = self.refresher.start(self.refreshInterval['s'], now=True)
        
    @inlineCallbacks
    def stopServer(self):
        """Kill the device refresh loop and wait for it to terminate."""
        if hasattr(self, 'refresher'):
            self.refresher.stop()
            yield self.refresherDone

    def readTemperatures(self):
        """Read temperatures from a log file."""
        # Get the list of files in the folder and return the one with
        # the most recent name.
        file = sorted([f for f in os.listdir(self._path)
                if os.path.isfile(os.path.join(self._path, f))])[-1]

        # Read the last line in the log file.
        with open(os.path.join(self._path, file), 'rb') as f:
            f.seek(0, os.SEEK_END)
            sz = f.tell()   # Get the size of the file.
            while True:
                if self._offset > sz:
                    self._offset = sz
                f.seek(-self._offset, os.SEEK_END)
                lines = f.readlines()
                if len(lines) > 1 or self._offset == sz:
                    line = lines[-1]
                    break
                self._offset *= 2
            # Extract temperatures.
            fields = line.split('\t')
            self._still_temp = float(fields[10]) * mK
            self._exchange_temp = float(fields[11]) * mK
            self._mix_temp = float(fields[12]) * mK
                
    @setting(1, 'Refresh Temperatures')
    def refresh_temperatures(self, c):
        """Manually refresh the temperatures."""
        self.readTemperatures()
        
    @setting(10, 'Still Temperature', returns='v[mK]')
    def still_temperature(self, c):
        """Return the still chamber temperature."""
        return self._still_temp

    @setting(11, 'Exchange Temperature', returns='v[mK]')
    def exchange_temperature(self, c):
        """Return the exchange chamber temperature."""
        return self._exchange_temp
        
    @setting(12, 'Mix Temperature', returns='v[mK]')
    def mix_temperature(self, c):
        """Return the mix chamber temperature."""
        return self._mix_temp


__server__ = LeidenDRPseudoserver()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)