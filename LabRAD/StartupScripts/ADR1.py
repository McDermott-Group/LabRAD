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
This script can be used to start ADR3 client with all necessary servers. 
"""
import sys
import subprocess as sp

def main():
    sp.Popen([sys.executable, 'electronics.py',
        '--registry-start-program-key', 'Start ADR1 Program List',
        '--registry-start-server-key', 'Start ADR1 Server List'],
        cwd='.')

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()