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
# along with this program.  If not, see <http://www.gnu.org/licenses/>
  
"""
### BEGIN NODE INFO
[info]
name = Heat Switch
version = 1.0
description = Provides a way to open and close the Heat Switch for the Shasta ADRs.
  
[startup]
cmdline = %PYTHON% %FILE%
timeout = 20
  
[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""
  
from labrad.server import setting, LabradServer
from twisted.internet.defer import inlineCallbacks, returnValue

class HeatSwitchServer(LabradServer):
    name = 'Heat Switch' # Server name
    deviceName = 'Heat Switch' # *IDN? doesnt work for this one, ugh
  
    @setting(11, 'Close')
    def close(self, c):
        """Closes the Heat Switch."""
        reg = self.client.registry
        yield reg.cd( c['adr settings path'] )
        addr = yield reg.get('Heat Switch Close Address')
        self.setAddr(addr)
        self._write('close') # can write anything as long as there is an up pulse (so not x00)
        
    @setting(12, 'Open')
    def open(self, c):
        """Opens the Heat Switch."""
        reg = self.client.registry
        yield reg.cd( c['adr settings path'] )
        addr = yield reg.get('Heat Switch Open Address')
        self.setAddr(addr)
        self._write('open') # can write anything as long as there is an up pulse (so not x00)
    
    def setAddr(self,addr):
        self.port = addr
    
    @inlineCallbacks
    def _write(self,data):
        """Since *IDN? does not work with this instrument, we cannot select devices or 
        write in the normal way.  This contacts all gpib buses and looks for the selected
        address and writes the data to it."""
        allServers = yield self.client.manager.servers()
        servers = [s for n, s in allServers
                     if (('Serial Server' in s) or ('serial_server' in s)) ]
        for serv in servers:
            devices = yield self.client[serv].list_serial_ports()
            if self.port in devices:
                yield self.client[serv].open(self.port)
                yield self.client[serv].write(data)
                yield self.client[serv].close()
    
    @setting(15,'Set ADR Settings Path',path=['*s'])
    def set_adr_settings_path(self,c,path):
        c['adr settings path'] = path
  
__server__ = HeatSwitchServer()
  
if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)