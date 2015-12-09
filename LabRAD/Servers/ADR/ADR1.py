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
name = ADR1
version = 1.3.2-no-refresh
description = This Labrad server controls the ADRs we have.  It can be connected to by ADRClient.py or other labrad clients to control the ADR with a GUI, etc.
instancename = ADR1

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

from ADRServer import *
import sys

if __name__ == "__main__":
    if '-a' not in sys.argv:
        sys.argv.append('-a')
        sys.argv.append('ADR1')
    __server__ = ADRServer(sys.argv)
    util.runServer(__server__)