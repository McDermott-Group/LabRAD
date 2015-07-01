# Copyright (C) 2007  Matthew Neeley
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
name = Agilent 6641A PS
version = 1.0
description = Power supply for the ADR magnet.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from labrad.server import setting, returnValue
import labrad.units as units

class AgilentPSServer(GPIBManagedServer):
    name = 'Agilent 6641A PS'
    deviceName = 'HEWLETT-PACKARD 6641A'
    deviceWrapper = GPIBDeviceWrapper
    
    @setting(10000, 'Output State', os=['b'], returns=['b'])
    def output_state(self, c, os=None):
        """Get or set the output state, on or off."""
        dev = self.selectedDevice(c)
        if os is None:
            resp = yield dev.query('OUTP?')
            os = bool(int(resp))
        else:
            print 'OUTP %i' % int(os)
            yield dev.write('OUTP %i' % int(os))
        returnValue(os)

    @setting(10001, 'Current', cur=['v[A]'], returns=['v[A]'])
    def current(self, c, cur=None):
        """Get or set the current."""
        dev = self.selectedDevice(c)
        if cur is None:
            resp = yield dev.query('MEAS:CURR?')
            cur = float(resp)*units.A
        else:
            yield dev.write('CURR %f' % cur['A'])
        returnValue(cur)

    @setting(10002, 'Voltage', v=['v[V]'], returns=['v[V]'])
    def voltage(self, c, v=None):
        """Get or set the voltage."""
        dev = self.selectedDevice(c)
        if v is None:
            resp = yield dev.query('MEAS:VOLT?')
            v = float(resp)*units.V
        else:
            yield dev.write('VOLT %f' % v['V'])
        returnValue(v)
    
    @setting(10003, 'Reset')
    def reset(self, c):
        """Reset the PS."""
        dev = self.selectedDevice(c)
        yield dev.write('*RST')
    
    @setting(10004, 'Get Operating Mode', returns=['s'])
    def getOpsReg(self, c):
        """Get the operating mode."""
        dev = self.selectedDevice(c)
        returnedVal = yield dev.query('STAT:OPER:COND?')#.strip('\x00'))
        opCode = int(returnedVal)
        codeList = [int(x) for x in "{0:016b}".format(opCode)]
        codeList.reverse()
        CAL = bool(codeList[0])
        WTG = bool(codeList[5])
        CV = bool(codeList[8])
        CC = bool(codeList[10])
        if CV: returnValue( 'CV Mode' )
        elif CC: returnValue( 'CC Mode' )
        else: returnValue( 'OutputOff' )
    
    @setting(10005, 'Initialize PS')
    def initialize(self, c, currentLimit=9.0*units.A):
        """Initialize the PS, keeping the current and voltage as they are.  The default current limit is 9A."""
        dev = self.selectedDevice(c)
        #current = yield self.current(c)
        cur = yield self.current(c)
        if cur-(0.01*units.A) >= currentLimit:
            message = 'Current too high! Manually lower before trying to run again. Please quit now.\n'
            #self.log.log(message, alert=True)
        else:
            state = yield self.getOpsReg(c)
            if state == 'OutputOff':
                message = 'Output Off. Setting Current to '+str(currentLimit)+' Amps and voltage to 0 Volts.\n'
                #self.log.log(message)
                yield self.reset(c)
                yield self.current(c,currentLimit)
                yield self.voltage(c,0*units.V)
                yield self.output_state(c,True)
            elif state == 'CV Mode':
                message = 'Starting in CV Mode. Setting Current to '+str(currentLimit)+' Amps.\n'
                #self.log.log(message)
                yield self.current(c, currentLimit )
            elif state == 'CC Mode':
                V_now = yield self.voltage(c)
                message = 'Starting in CC Mode. Setting Current to '+str(currentLimit)+' Amps and voltage to '+str(V_now)+' Volts.\n'
                #self.log.log(message)
                yield self.voltage(c, V_now )
                yield self.current(c, currentLimit )

__server__ = AgilentPSServer()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
