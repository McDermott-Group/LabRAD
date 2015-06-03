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
name = SIM900 SRS Mainframe
version = 1.3.2-no-refresh
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

from labrad.server import setting, returnValue
from twisted.internet.defer import inlineCallbacks, returnValue
#from twisted.internet.reactor import callLater
#from twisted.internet.task import LoopingCall
from gpib_server import GPIBBusServer
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
import visa
import sys, types

if sys.version_info >= (2,7): py27,py26 = True,False #python2.7
else: py27,py26 = False,True #python2.6
try:
    if float(visa.__version__) < 1.6: v15,v17= True,False
    else: v15,v17 = False,True
    if v15: from visa import vpp43
except Exception as e: v15,v17= True,False

# These are the new functions that will be used for the visa instruments to connect to the appropriate slots
def setSlot(self,i):
    self.slot = i
def getSlot(self):
    return self.slot
def newRead(self,*args,**kwargs):
    self.oldWrite("CONN %d,'xyz'" %self.slot)
    response = self.oldRead(*args,**kwargs)
    self.oldWrite("xyz")
    return response
def newWrite(self,*args,**kwargs):
    self.oldWrite("CONN %d,'xyz'" %self.slot)
    response = self.oldWrite(*args,**kwargs)
    self.oldWrite("xyz")
    return response
def newAsk(self,*args,**kwargs):
    self.oldWrite("CONN %d,'xyz'" %self.slot)
    response = self.oldAsk(*args,**kwargs)
    self.oldWrite("xyz")
    return response
    

class SIM900Server(GPIBBusServer,GPIBManagedServer):#,object):
    name = '%LABRADNODE% SIM900'
    deviceName = 'Stanford_Research_Systems SIM900'
    deviceWrapper = GPIBDeviceWrapper
    
    def __init__(self):
        GPIBManagedServer.__init__(self)
        GPIBBusServer.__init__(self)
    
    @inlineCallbacks
    def initServer(self):
        """Provides direct access to GPIB-enabled devices in a SIM900 mainframe."""
        GPIBManagedServer.initServer(self)
        GPIBBusServer.initServer(self)
        #subscribe to messages
        # connect_func = lambda c, (s, payload): self.refreshDevices()
        # disconnect_func = lambda c, (s, payload): self.refreshDevices()
        # mgr = self.client.manager
        # self._cxn.addListener(connect_func, source=mgr.ID, ID=10)
        # self._cxn.addListener(disconnect_func, source=mgr.ID, ID=11)
        # yield mgr.subscribe_to_named_message('GPIB Device Connect', 10, True)
        # yield mgr.subscribe_to_named_message('GPIB Device Disconnect', 11, True)
        p = yield self.client.gpib_device_manager.packet()
        p.register_ident_function( 'custom_ident_function' ) #['custom_ident_function',210] )
        result = yield p.send()
        
    @inlineCallbacks
    def refreshDevices(self):
        """Refresh the list of known devices (modules) in the SIM900 mainframe."""
        try:
            addresses = []
            IDs, names = self.deviceLists()
            for _,SIM900addr in zip(IDs, names):
                try:
                    p = yield self.client[self.name].packet()
                    p.select_device(SIM900addr)
                    p.gpib_query('CTCR?')
                    res = yield p.send()
                    statusStr = res['gpib_query']
                except Exception, e: print 'this is the error',e
                # ask the SIM900 which slots have an active module, and only deal with those.
                statusCodes = [bool(int(x)) for x in "{0:016b}".format(int(statusStr))]
                statusCodes.reverse()
                for i in range(1,9): #slots 1-8 in rack
                    if statusCodes[i]: #added or changed
                        # Ex: mcdermott5125 GPIB Bus - GPIB0::2::SIM900::4
                        devName = SIM900addr.split(' - ')[-1]+'::SIM900::'+str(i)
                        addresses.append(devName)
            additions = set(addresses) - set(self.mydevices.keys())
            deletions = set(self.mydevices.keys()) - set(addresses)
            # get the visa instruments, changing the read/write/ask commands to work for only the correct slot in the SIM900
            for addr in additions:
                try:
                    instName = addr.split(' - ')[-1].rsplit('::',2)[0]
                    if py27: 
                        rm = visa.ResourceManager()
                        instr = rm.open_resource(instName, open_timeout=1.0)#python2.7
                    else: instr = visa.instrument(instName, timeout=1.0, term_chars='')
                    #change the read, write, ask settings to automatically go to right module in SIM rack
                    instr.oldRead, instr.oldWrite, instr.oldAsk = instr.read, instr.write, instr.ask
                    instr.setSlot = setSlot
                    instr.getSlot = getSlot
                    instr.read = types.MethodType(newRead,instr)
                    instr.write = types.MethodType(newWrite,instr)
                    instr.ask = types.MethodType(newAsk,instr)
                    instr.setSlot(instr,int(addr[-1]))
                    #instr.clear()
                    self.mydevices[addr] = instr
                    self.sendDeviceMessage('GPIB Device Connect', addr)
                except Exception, e:
                    print 'Failed to add ' + addr + ':' + str(e)
            for addr in deletions:
                del self.mydevices[addr]
                self.sendDeviceMessage('GPIB Device Disconnect', addr)
        except Exception, e:
            print 'Problem while refreshing devices:', str(e)
    
    @setting(210, server='s', addr='s', idn='s', returns='s')
    def custom_ident_function(self, c, server, addr, idn=None):
        return self.mydevices[addr].ask('*IDN?')


__server__ = SIM900Server()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
