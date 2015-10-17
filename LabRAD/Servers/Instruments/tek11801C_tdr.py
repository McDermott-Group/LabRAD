# Created by Alexander Opremcak, 10/17/2015
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
name = Tekronix 11801C Digital Sampling Scope
version = 1.0
description = Basic Functionality for TDR
  
[startup]
cmdline = %PYTHON% %FILE%
timeout = 20
  
[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

#does message number matter?

from labrad.server import setting
from labrad.gpib import GPIBManagedServer, GPIBDeviceWrapper
from twisted.internet.defer import inlineCallbacks, returnValue
from labrad import units
import numpy as np
        
class Tek11801C_Server(GPIBManagedServer):
    name = 'Tek11801C' # Server name
    deviceName = ['ID TEK/11801C']
  
    @setting(10, 'getInstrName', returns='s')
    def getInstrumentName(self, c):
        """Returns name."""
        dev = yield self.selectedDevice(c)
        instrumentName = yield dev.query('ID?')
        returnValue(instrumentName)

    @setting(11, 'getNumSamplingHeads', returns='i')
    def getNumSamplingHeads(self, c):
        """Returns the number of sampling heads recognized by the Tek11801C."""
        dev = yield self.selectedDevice(c)
        resp = yield dev.query('ACQNUM?') # returns 'ACQNUM 2' for example
        print resp
        print resp.split(' ')
        NumAcqSys= int(resp.split(" ")[1])
        returnValue(NumAcqSys)

    @setting(12, 'getAveragingState', returns='s')
    def getAveragingState(self, c):
        """Returns the averaging state of the Tek11801C."""
        dev = yield self.selectedDevice(c)
        resp = yield dev.query('AVG?') # returns 'AVG ON' or 'AVG OFF'
        returnValue(resp.split(" ")[1]) #parsed to return either on or off

    @setting(13, 'setAveragingState',avgState='s', returns='')
    def setAveragingState(self, c, avgState):
        """Sets the averaging state of the Tek11801C."""
        dev = yield self.selectedDevice(c)
        possibilities = ['ON', 'OFF']
        if avgState.upper() in (pos.upper() for pos in possibilities):
            dev.write('AVG'+' '+avgState.upper())
        else:
            print 'Acceptable inputs for this function are \'ON\' or \'OFF\''
        return

    @setting(14, 'getChanVoltOffset',chanNum='w', returns='v[Volts]')
    def getChanVoltOffset(self, c, chanNum):
        """Returns the averaging state of the Tek11801C."""
        dev = yield self.selectedDevice(c)
        resp = yield dev.query('CHM'+str(chanNum)+'?'+' OFFSET') # returns 'AVG ON' or 'AVG OFF'
        print resp.split(':')[1]
        returnValue(float(resp.split(':')[1])) #parsed to return either on or off
        

    @setting(27, 'Get Num Points', returns='s')
    def getNumTracePoints(self, c):
        """Returns number of points."""
        dev = self.selectedDevice(c)
        length = yield dev.query('TBM? LEN')
        returnValue(length)

    @setting(28, 'Get Trace Data', traceNum='w', returns='(s*vs*v)') 
    def getTraceData(self, c, traceNum):
        """Returns trace data"""
        dev = self.selectedDevice(c)
        #if traceNum is None: # assume user wants all traces
        #    traceData = yield dev.query('OUTPUT ALLTRACE;WAVfrm?')
        if traceNum >=1: # user specifies 
            traceData = yield dev.query('OUTPUT TRACE'+str(traceNum)+';WAVfrm?')
        splitter = traceData.split(';')
        for ii in range(0,len(splitter),2): # increment by 2's 1 preamble & 1 data string per trace
            tracePreamble = splitter[ii].split(',')
            traceDataStr = splitter[ii+1].split(',')
            for jj in range(0, len(tracePreamble)):
                if 'XINCR' in tracePreamble[jj]:
                    XINCR = float(tracePreamble[jj].split(':')[1])
                elif 'XMULT' in tracePreamble[jj]:
                    XMULT = float(tracePreamble[jj].split(':')[1])
                elif 'XUNIT' in tracePreamble[jj]:
                    XUNIT = tracePreamble[jj].split(':')[1]
                elif 'XZERO' in tracePreamble[jj]:
                    XZERO = float(tracePreamble[jj].split(':')[1])
                elif 'YMULT' in tracePreamble[jj]:
                    YMULT = float(tracePreamble[jj].split(':')[1])
                elif 'YUNIT' in tracePreamble[jj]:
                    YUNIT = tracePreamble[jj].split(':')[1]
                elif 'YZERO' in tracePreamble[jj]:
                    YZERO = float(tracePreamble[jj].split(':')[1])
            tdrYData = np.empty(len(traceDataStr))
            tdrXData = np.empty(len(traceDataStr))
            for kk in range (1, len(traceDataStr)):
                tdrYData[kk-1] = (YZERO+YMULT*float(traceDataStr[kk]))
                tdrXData[kk-1] = (XINCR*(kk-1)+XZERO)
        returnValue([XUNIT, tdrXData, YUNIT, tdrYData])
    
  
__server__ = Tek11801C_Server()
  
if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
