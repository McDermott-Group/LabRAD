# Copyright (C) 2015 Chris Wilen, Ivan Pechenezhskiy 
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
name = SIM900 SRS Mainframe
version = 2.0
description = Gives access to SIM900, GPIB devices in it.
instancename = %LABRADNODE% SIM900

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

import os.path, sys
SCRIPT_PATH = os.path.dirname(__file__)
GPIB_PATH = os.path.join(SCRIPT_PATH.split('LabRAD')[0], 'LabRAD\Servers\GPIB')
if GPIB_PATH not in sys.path:
    sys.path.append(GPIB_PATH)

from types import MethodType
import visa

from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet.reactor import callLater
# from twisted.internet.task import LoopingCall

from labrad.server import setting, returnValue
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper

from gpib_server import GPIBBusServer
from gpib_device_manager import UNKNOWN

# These are the redefined functions that will be used for the visa instruments to connect to the appropriate slots.
def set_slot(self, i):
    self.slot = i

def get_slot(self):
    return self.slot

def read_decorated(self, *args):
    self.write_undecorated("CONN " + str(self.slot) + ",'XyZ'")
    try:
        response = self.read_undecorated(*args)
    except:
        response = None
        print("No response from " + str(self.address))
    self.write_undecorated('XyZ')
    return response

def read_raw_decorated(self, *args):
    self.write_undecorated("CONN " + str(self.slot) + ",'xzY'")
    try:
        response = self.read_undecoratedRaw(*args)
    except:
        response = None
        print("No response from " + str(self.address))
    self.write_undecorated('xzY')
    return response

def write_decorated(self, *args):
    self.write_undecorated("CONN " + str(self.slot) + ",'xZy'")
    response = self.write_undecorated(*args)
    self.write_undecorated('xZy')
    return response

def query_decorated(self, *args):
    self.write_undecorated("CONN " + str(self.slot) + ",'yXz'")
    try:
        response = self.query_undecorated(*args)
    except:
        response = None
        print("No response from " + str(self.address) + " to '" + str(args[0]) + "'")
    self.write_undecorated('yXz')
    return response    

class SIM900Server(GPIBBusServer, GPIBManagedServer):
    name = '%LABRADNODE% SIM900'
    deviceName = 'STANFORD RESEARCH SYSTEMS SIM900'
    deviceWrapper = GPIBDeviceWrapper
    
    def __init__(self):
        GPIBBusServer.__init__(self)
        GPIBManagedServer.__init__(self)
    
    @inlineCallbacks
    def initServer(self):
        """Provides direct access to GPIB-enabled devices in a SIM900 mainframe."""
        GPIBBusServer.initServer(self)
        GPIBManagedServer.initServer(self)
        p = yield self.client.gpib_device_manager.packet()
        p.register_ident_function('custom_ident_function')
        yield p.send()
    
    @inlineCallbacks
    def handleDeviceMessage(self,*args):
        oldDevices = self.devices.copy()
        yield GPIBManagedServer.handleDeviceMessage(self,*args)
        if self.devices != oldDevices:
            callLater(0.1, self.refreshDevices)     # To improve the responsiveness of the SIM900 device to GPIB queries.
        
    @inlineCallbacks
    def refreshDevices(self):
        """Refresh the list of known devices (modules) in the SIM900 mainframe."""
        yield self.client.refresh()     # To avoid calling before the server name was added to the client dictionary.
        addresses = []
        statusStr = '0'
        IDs, names = self.deviceLists()
        for SIM900addr in names:
            p = yield self.client[self.name].packet()
            res = yield p.select_device(SIM900addr).gpib_write('*CLS').gpib_query('CTCR?').send()
            statusStr = res['gpib_query']
            # ask the SIM900 which slots have an active module, and only deal with those.
            statusCodes = [bool(int(x)) for x in "{0:016b}".format(int(statusStr))]
            statusCodes.reverse()
            for i in range(1, 9): # slots 1-8 in rack
                # print('Device on ' + SIM900addr.split(' - ')[-1] + ' in slot ' + str(i) + '?: ' + str(statusCodes[i]))
                if statusCodes[i]: # added or changed
                    # Ex: mcdermott5125 GPIB Bus - GPIB0::2::SIM900::4
                    devName = '::'.join(SIM900addr.split(' - ')[-1].split('::')[:-1] + ['SIM900', str(i)])
                    addresses.append(devName)
        additions = set(addresses) - set(self.mydevices.keys())
        deletions = set(self.mydevices.keys()) - set(addresses)
        # Get the visa instruments, changing the read/write/query commands to work for only the correct slot in the SIM900.
        for addr in additions:
            instName = addr.split(' - ')[-1].rsplit('::', 2)[0]
            rm = visa.ResourceManager()
            instr = rm.open_resource(instName, open_timeout=1.0)
            instr.write_termination = ''
            # Change (decorate) the read, write, query settings to automatically go to right module in SIM rack.
            instr.read_undecorated, instr.read_undecorated = instr.read, instr.read_raw 
            instr.write_undecorated, instr.query_undecorated = instr.write, instr.query
            instr.set_slot = set_slot
            instr.get_slot = get_slot
            instr.read = MethodType(read_decorated, instr)
            instr.read_raw = MethodType(read_raw_decorated, instr)
            instr.write = MethodType(write_decorated, instr)
            instr.query = MethodType(query_decorated, instr)
            instr.set_slot(instr, int(addr[-1]))
            instr.address = addr
            self.mydevices[addr] = instr
            self.sendDeviceMessage('GPIB Device Connect', addr)
        for addr in deletions:
            del self.mydevices[addr]
            self.sendDeviceMessage('GPIB Device Disconnect', addr)
  
    @setting(210, server='s', addr='s', idn='s', returns='?')
    def custom_ident_function(self, c, server, addr, idn=None):
        if addr in self.mydevices:
            yield self.mydevices[addr].write('*CLS')
            res = yield self.mydevices[addr].query('*IDN?')
            if res is not None:
                mfr, model, ver, rev = res.upper().split(',')
                returnValue(mfr.replace('_', ' ') + ' ' + model)

__server__ = SIM900Server()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)