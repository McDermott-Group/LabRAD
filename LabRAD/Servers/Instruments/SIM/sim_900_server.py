# Copyright (C) 2008  Matthew Neeley
#           (C) 2015  Chris Wilen, Ivan Pechenezhskiy 
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
name = SIM900
version = 1.4.1
description = Gives access to GPIB devices in the SIM900 mainframe.
instancename = SIM900

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.reactor import callLater

from labrad.server import LabradServer, setting
from labrad.errors import DeviceNotSelectedError
import labrad.units as units
from labrad.gpib import GPIBManagedServer

import string

class SIM900(GPIBManagedServer):
    """Provides direct access to GPIB-enabled devices."""
    name = 'SIM900'
    deviceName = 'STANFORD RESEARCH SYSTEMS SIM900'
    # refreshInterval = 10
    defaultTimeout = 1.0*units.s

    def initServer(self):
        GPIBManagedServer.initServer(self)
        self.mydevices = {}
        # start refreshing only after we have started serving
        # this ensures that we are added to the list of available
        # servers before we start sending messages
        callLater(0.1, self.refreshDevices)

    @inlineCallbacks
    def handleDeviceMessage(self, *args):
        """We override this function so that whenever a new SIM900 is
        added, and a message is sent out, we refresh the devices. This
        has the benefit of being able to start this server, the 
        GPIB Device Manager, and the GPIB Bus Server, in any order."""
        yield GPIBManagedServer.handleDeviceMessage(self, *args)
        if args[0] == self.deviceName: 
            self.refreshDevices()
    
    @inlineCallbacks
    def refreshDevices(self):
        """
        Refresh the list of known devices (modules) in the SIM900
        mainframe.
        """
        print('Refreshing devices...')
        addresses = []
        IDs, names = self.deviceLists()
        for SIM900addr in names:
            p = self.client[self.name].packet()
            res = yield p.select_device(SIM900addr).gpib_write('*RST').gpib_write('*CLS').gpib_query('CTCR?').send()
            statusStr = res['gpib_query']
            # Ask the SIM900 which slots have an active module, and only deal with those.
            statusCodes = [bool(int(x)) for x in "{0:016b}".format(int(statusStr))]
            statusCodes.reverse()
            for i in range(1, 9): # slots 1-8 in rack
                if statusCodes[i]: # added or changed
                    # Ex: mcdermott5125 GPIB Bus - GPIB0::2[::INSTR]::SIM900::4
                    devName = ('::'.join(SIM900addr.split(' - ')[-1].split('::')[:-1] + 
                            ['SIM900', str(i)]))
                    devName = SIM900addr+'::SIM900::'+str(i)
                    addresses.append(devName)
        additions = set(addresses) - set(self.mydevices.keys())
        deletions = set(self.mydevices.keys()) - set(addresses)
        # Get the visa instruments, changing the read/write/query
        # commands to work for only the correct slot in the SIM900.
        for addr in additions:
            instName = addr.split(' - ')[-1].rsplit('::', 2)[0]
            self.mydevices[addr] = instName
            self.sendDeviceMessage('GPIB Device Connect', addr)
        for addr in deletions:
            del self.mydevices[addr]
            self.sendDeviceMessage('GPIB Device Disconnect', addr)

    def getSocketsList(self):
        """Get a list of all connected devices.

        Return value:
        A list of strings with the names of all connected devices, ready
        for being used to open each of them.
        """
        return self.rm.list_resources()

    def sendDeviceMessage(self, msg, addr):
        print(msg + ': ' + addr)
        self.client.manager.send_named_message(msg, (self.name, addr))

    def initContext(self, c):
        c['timeout'] = self.defaultTimeout

    @setting(19, returns='*s')
    def list_addresses(self, c):
        """Get a list of GPIB addresses on this bus."""
        return sorted(self.mydevices.keys())

    @setting(21)
    def refresh_devices(self, c):
        '''Manually refresh devices.'''
        self.refreshDevices()
 
    @setting(20, addr='s', returns='s')
    def address(self, c, addr=None):
        """Get or set the GPIB address for this context.

        To get the addresses of available devices,
        use the list_devices function.
        """
        if addr is not None:
            c['addr'] = addr
        return c['addr']

    @setting(22, time='v[s]', returns='v[s]')
    def timeout(self, c, time=None):
        """Get or set the GPIB timeout."""
        if time is not None:
            c['timeout'] = time
        return c['timeout'] 
  
    @setting(23, data='s', returns='')
    def write(self, c, data):
        """Write a string to the GPIB bus."""
        if 'addr' not in c:
            raise DeviceNotSelectedError("No GPIB address selected")
        if c['addr'] not in self.mydevices:
            raise Exception('Could not find device ' + c['addr'])
        # Ex: mcdermott5125 GPIB Bus - GPIB0::2::SIM900::4
        gpibBusServName = c['addr'].split(' - ')[0]
        slot = c['addr'][-1]
        p = self.client[gpibBusServName].packet()
        p.address(self.mydevices[c['addr']])
        p.timeout(c['timeout'])
        p.write("CONN " + str(slot) + ",'xZy'")
        p.write(data)
        p.write('xZy')
        p.send()

    @setting(24, bytes='w', returns='s')
    def read_raw(self, c, bytes=None):
        """Read a raw string from the GPIB bus.

        If specified, reads only the given number of bytes.
        Otherwise, reads until the device stops sending.
        """
        if 'addr' not in c:
            raise DeviceNotSelectedError("No GPIB address selected")
        if c['addr'] not in self.mydevices:
            raise Exception('Could not find device ' + c['addr'])
        # Ex: mcdermott5125 GPIB Bus - GPIB0::2::SIM900::4
        gpibBusServName = c['addr'].split(' - ')[0]
        slot = c['addr'][-1]
        p = self.client[gpibBusServName].packet()
        p.address(self.mydevices[c['addr']])
        p.timeout(c['timeout'])
        p.write("CONN " + str(slot) + ",'xZy'")
        p.read_raw(bytes)
        p.write('xZy')
        resp = yield p.send()
        returnValue(resp['read_raw'])

    @setting(25, returns='s')
    def read(self, c):
        """Read from the GPIB bus."""
        if 'addr' not in c:
            raise DeviceNotSelectedError("No GPIB address selected")
        if c['addr'] not in self.mydevices:
            raise Exception('Could not find device ' + c['addr'])
        # Ex: mcdermott5125 GPIB Bus - GPIB0::2::INSTR::SIM900::4
        gpibBusServName = c['addr'].split(' - ')[0]
        slot = c['addr'][-1]
        p = self.client[gpibBusServName].packet()
        p.address(self.mydevices[c['addr']])
        p.timeout(c['timeout'])
        p.write("CONN " + str(slot) + ",'xZy'")
        p.read()
        p.write('xZy')
        resp = yield p.send()
        returnValue(resp['read'])

    @setting(26, data='s', returns='s')
    def query(self, c, data):
        """Make a GPIB query.

        This query is atomic. No other communication to the
        device will occur while the query is in progress.
        """
        if 'addr' not in c:
            raise DeviceNotSelectedError("No GPIB address selected")
        if c['addr'] not in self.mydevices:
            raise Exception('Could not find device ' + c['addr'])
        # Ex: mcdermott5125 GPIB Bus - GPIB0::2[::INSTR]::SIM900::4
        gpibBusServName = c['addr'].split(' - ')[0]
        slot = c['addr'][-1]
        p = self.client[gpibBusServName].packet()
        p.address(self.mydevices[c['addr']])
        p.timeout(c['timeout'])
        p.write("CONN " + str(slot) + ",'xZy'")
        p.query(data)
        p.write('xZy')
        resp = yield p.send()
        returnValue(resp['query'])

__server__ = SIM900()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)