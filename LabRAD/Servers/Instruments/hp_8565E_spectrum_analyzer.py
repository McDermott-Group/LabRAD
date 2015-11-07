# Copyright (C) 2015 Guilhem Ribeill
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
name = HP 8565E Spectrum Analyzer
version = 0.9
description = HP8655E Spectrum Analyzer

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import os
if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    # This is executed when the script is loaded by the labradnode.
    SCRIPT_PATH = os.path.dirname(os.getcwd())
else:
    # This is executed if the script is started by clicking or
    # from a command line.
    SCRIPT_PATH = os.path.dirname(__file__)
LABRAD_PATH = os.path.join(SCRIPT_PATH.rsplit('LabRAD', 1)[0])
import sys
if LABRAD_PATH not in sys.path:
    sys.path.append(LABRAD_PATH)

import numpy

from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from labrad.server import setting, returnValue
import labrad.units as units

from LabRAD.Servers.Utilities.general import sleep


class Agilent8720ETServer(GPIBManagedServer):
    name = 'HP 8565E Spectrum Analyzer'
    deviceName = 'HEWLETT PACKARD 8565E'
    deviceWrapper = GPIBDeviceWrapper
    
    @setting(551, 'Preset')
    def preset(self, c):
        """Reset the network analyzer."""
        dev = self.selectedDevice(c)
        yield dev.write('IP')
        
    @setting(552, 'Center Frequency', freq=['v[Hz]'], returns=['v[Hz]'])
    def center_frequency(self, c, freq=None):
        """Set or get the center frequency of the trace."""
        dev = self.selectedDevice(c)
        if freq is None:
            resp = yield dev.query('CF?')
            freq = float(resp) * units.Hz
        else:
            yield dev.write('CF %i' %freq['Hz'])
        returnValue(freq)

    @setting(949, 'Initialize')
    def initialize(self, c):
        """Initialize the network analyzer."""
        dev = self.selectedDevice(c)
        yield self.preset(c)
        print('Initialized...')


__server__ = Agilent8720ETServer()


if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)