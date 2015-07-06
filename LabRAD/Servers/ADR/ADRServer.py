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
name = ADR Server
version = 1.3.2-no-refresh
description = This Labrad server controls the ADRs we have.  It can be connected to by ADRClient.py or other labrad clients to control the ADR with a GUI, etc.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""
ADR_SETTINGS_BASE_PATH = ['','ADR Settings'] # path in registry
DEFAULT_ADR = 'ADR3' # name of ADR in registry
AVAILABLE_ADRS = ['ADR1','ADR2','ADR3']

import matplotlib as mpl
import numpy, pylab
import datetime, struct
from labrad.server import (LabradServer, setting,
                           inlineCallbacks, returnValue)
from labrad.devices import DeviceServer
from labrad import util, units
from labrad.types import Error as LRError
import sys
 
def deltaT(dT):
    """.total_seconds() is only supported by >py27 :(, so we use this to subtract two datetime objects."""
    return dT.days*86400 + dT.seconds + dT.microseconds*pow(10,-6)

class ADRServer(DeviceServer):
    """Provides a way to control all the instruments that control our ADRs."""
    name = 'ADR Server'
    deviceName = 'ADR'
    # We no longer use signals.  That way if this server is turned on and off, named_messages still get to clients.  This is an example of a signal, however:
    # stateChanged = Signal(1001, 'signal:state_changed', 's')
    
    def __init__(self, args):
        DeviceServer.__init__(self)
        self.ADRSettingsPath = ADR_SETTINGS_BASE_PATH
        selectedADR = DEFAULT_ADR
        if '-a' in args:    # Use -a to specify ADR
            index = args.index('-a')
            args.pop(index)
            selection = str( args.pop(index) )   # if we do not pop these off, twisted will complain because this is not an allowed argument
            if selection in AVAILABLE_ADRS:
                selectedADR = selection
            else: 
                print '%s is not a valid ADR selection.' %selection
        self.ADRSettingsPath.append(selectedADR)
        self.name = selectedADR
        self.deviceName = selectedADR
        print '%s selected.' %selectedADR
        self.alive = True
        self.state = {  'T_FAA': numpy.NaN*units.K,
                        'T_GGG': numpy.NaN*units.K,
                        'T_3K' : numpy.NaN*units.K,
                        'T_60K': numpy.NaN*units.K,
                        'datetime' : datetime.datetime.now(),
                        'cycle': 0,
                        'magnetV': numpy.NaN*units.V,
                        'RuOxChan':'FAA',
                        'RuOxChanSetTime':datetime.datetime.now(),
                        'PSCurrent':numpy.NaN*units.A,
                        'PSVoltage':numpy.NaN*units.V,
                        'maggingUp':False,
                        'regulating':False,
                        'regulationTemp':0.1,
                        'PID_cumulativeError':0}
        self.lastState = self.state.copy()
        self.ADRSettings ={ 'PID_KP':2,
                            'PID_KI':0,
                            'PID_KD':70,
                            'magup_dV': 0.003,               #[V/step] How much do we increase the voltage by every second when maggin up? HPD Manual uses 10mV=0.01V, 2.5V/30min=1.4mV/s ==> Let's use a middle rate of 3mV/step. (1 step is about 1s)
                            'magnet_voltage_limit': 0.1,      #Back EMF limit in Volts
                            'current_limit': 9,               #Max Current in Amps
                            'voltage_limit': 2,               #Max Voltage in Volts.  At 9A, we usually get about 2.5-2.7V or 1.69V (with or without the external diode protection box), so this shouldn't need to be more than 3 or 2
                            'dVdT_limit': 0.008,              #Keep dV/dt to under this value [V/s]
                            'dIdt_magup_limit': 9./(30*60),   #limit on the rate at which we allow current to increase in amps/s (we want 9A over 30 min)
                            'dIdt_regulate_limit': 9./(40*60),#limit on the rate at which we allow current to change in amps/s (we want 9A over 40 min)
                            'step_length': 1.0,              #How long is each regulation/mag up cycle in seconds.  **Never set this less than 1.0sec.**  The SRS SIM922 only measures once a second and this would cause runaway voltages/currents.
                            'magnet_max_temp': 5,
                            'Power Supply':['Agilent 6641A PS','addr'],
                            'Ruox Temperature Monitor':['SIM921 Server','addr'],
                            'Diode Temperature Monitor':['SIM922 Server','addr'],
                            'Magnet Voltage Monitor':['SIM922 Server','addr']}
        self.instruments = {'Power Supply':None,
                            'Ruox Temperature Monitor':None,
                            'Diode Temperature Monitor':None,
                            'Magnet Voltage Monitor':None}
        dt = datetime.datetime.now()
        self.dateAppend = dt.strftime("_%y%m%d_%H%M")
        self.logMessages = []
    @inlineCallbacks
    def initServer(self):
        """This method loads default settings from the registry, starts servers and sets up instruments, and sets up listeners for GPIB device connect/disconnect messages."""
        DeviceServer.initServer(self)
        yield self.loadDefaults()
        yield self.startServers()
        yield util.wakeupCall( 2 ) # on the round ADR, the HP DMM takes forever to initialize.  This prevents it from going on before it is ready.
        yield self.initializeInstruments()
        #subscribe to messages
        connect_func = lambda c, (s, payload): self.gpib_device_connect(*payload)
        disconnect_func = lambda c, (s, payload): self.gpib_device_disconnect(*payload)
        mgr = self.client.manager
        self._cxn.addListener(connect_func, source=mgr.ID, ID=10)
        self._cxn.addListener(disconnect_func, source=mgr.ID, ID=11)
        yield mgr.subscribe_to_named_message('GPIB Device Connect', 10, True)
        yield mgr.subscribe_to_named_message('GPIB Device Disconnect', 11, True)
        self.updateState()
    @inlineCallbacks
    def loadDefaults(self):
        reg = self.client.registry
        yield reg.cd(self.ADRSettingsPath)
        _,settingsList = yield reg.dir()
        for setting in settingsList:
            self.ADRSettings[setting] = yield reg.get(setting)
    @inlineCallbacks
    def startServers(self):
        """This method just starts any of the necessary servers if they are not running."""
        #get the labrad node name and the servers that are already running
        runningServList = yield self.client.manager.servers()
        running_servers = [name for _,name in runningServList]
        for name in running_servers:
            if name.find('node ') >= 0: nodeName = name.strip('node ')
            else:
                import os
                nodeName = os.environ['COMPUTERNAME'].lower()
        #which do we need to start?
        reg = self.client.registry
        yield reg.cd(self.ADRSettingsPath)
        servList = yield reg.get('Start Server List')
        #go through and start all the servers that are not already running
        for server in servList:
            if server not in running_servers:
                try:
                    yield self.client.servers['node '+nodeName].start(server)
                    self.logMessage( server+' started.')
                except Exception as e:
                    self.logMessage( 'ERROR starting '+server+': '+str(e) ,alert=True)
            else: self.logMessage(server+' is already running.')
    @inlineCallbacks
    def initializeInstruments(self):
        """This method simply creates the instances of the power supply, sim922, and ruox temperature monitor."""
        for instrName in self.instruments:
            settings = self.ADRSettings[instrName]
            try:
                instr = self.client[ settings[0] ]
                self.instruments[instrName] = instr
                try: yield instr.set_adr_settings_path(self.ADRSettingsPath)
                except: pass
                yield instr.select_device( settings[1] )
                if hasattr(instr,'connected'):
                    if instr.connected == False:
                        self.logMessage(instrName+' Connected.')
                else:
                    instr.connected = True
                    self.logMessage(instrName+' Connected.')
            except LRError as e:
                if 'NoDevicesAvailableError' in e.msg:
                    message = 'No devices connected for '+instrName+'.'
                    self.logMessage(message, alert=True)
                    instr.connected = False
                if 'NoSuchDeviceError' in e.msg:
                    message = 'No devices found for '+instrName+' at address '+settings[1]+'.'
                    self.logMessage(message, alert=True)
                    instr.connected = False
            except Exception as e:
                message = 'Could not connect to '+instrName+'. Check that it is turned on and the server is running.' + str(e)
                self.logMessage(message, alert=True)
                instr.connected = False
        try: 
            yield self.instruments['Power Supply'].initialize_ps()
            self.logMessage('Power Supply Initialized.')
        except Exception as e:
            self.logMessage( 'Power Supply could not be initialized.', alert=True)
        
    @inlineCallbacks
    def _refreshInstruments(self):
        """We can manually have all gpib buses refresh the list of devices connected to them."""
        self.logMessage('Refreshing Devices...')
        serverList = yield self.client.manager.servers()
        for serv in [n for s,n in serverList]:#[tuple[1].replace(' ','_').lower() for tuple in serverList]:
            if 'gpib_bus' in serv or 'GPIB Bus' in serv:# or 'sim900_srs_mainframe' in serv:
                yield self.client[serv].refresh_devices()
    def gpib_device_connect(self, server, channel):
        self.initializeInstruments()
    def gpib_device_disconnect(self, server, channel):
        self.initializeInstruments()
    @inlineCallbacks
    def logMessage(self, message, alert=False):
        """Applies a time stamp to the message and saves it to a file and an array."""
        dt = datetime.datetime.now()
        messageWithTimeStamp = dt.strftime("[%m/%d/%y %H:%M:%S] ") + message
        self.logMessages.append( (messageWithTimeStamp,alert) )
        try:
            yield self.client.registry.cd(self.ADRSettingsPath)
            file_path = yield self.client.registry.get('Log Path')
        except Exception as e:
            message = '{Saving log failed.  Check that AFS is working.} ' + message
        with open(file_path+'\\log'+self.dateAppend+'.txt', 'a') as f:
            f.write( messageWithTimeStamp + '\n' )
        print '[log] '+ message
        self.client.manager.send_named_message('Log Changed', (messageWithTimeStamp,alert))
    @inlineCallbacks
    def updateState(self):
        """ This takes care of the real time reading of the instruments. It starts immediately upon starting the program, and never stops. """
        nan = numpy.nan
        while self.alive:
            cycleStartTime = datetime.datetime.now()
            # update system state
            self.lastState = self.state.copy()
            try:
                self.state['T_60K'],self.state['T_3K'] = yield self.instruments['Diode Temperature Monitor'].get_diode_temperatures()
            except Exception as e: 
                self.state['T_60K'],self.state['T_3K'] = nan*units.K, nan*units.K
                self.instruments['Diode Temperature Monitor'].connected = False
            try:
                ruoxDevSettings = yield self.client.manager.lr_settings(self.ADRSettings['Ruox Temperature Monitor'][0])
                if 'Get Time Constant' in [n for s,n in ruoxDevSettings]:
                    timeConst = yield self.instruments['Ruox Temperature Monitor'].get_time_constant()
                else: timeConst = 0*units.s
                if deltaT( datetime.datetime.now() - self.state['RuOxChanSetTime'] ) >= 10*timeConst['s']: #only if we have waited 10 x the time constant for the reader to settle
                    self.state[ 'T_'+self.state['RuOxChan'] ] = yield self.instruments['Ruox Temperature Monitor'].get_ruox_temperature()
                    #print self.state['T_GGG']['K'], self.state['T_FAA']['K'], self.instruments['Ruox Temperature Monitor'], self.state['RuOxChanSetTime']
                    if self.state['RuOxChan'] == 'GGG': self.state['T_FAA'] = nan*units.K
                    if self.state['RuOxChan'] == 'FAA': self.state['T_GGG'] = nan*units.K
                    if self.state['T_GGG']['K'] == 20.0: self.state['T_GGG'] = nan*units.K
                    if self.state['T_FAA']['K'] == 45.0: self.state['T_FAA'] = nan*units.K
                    # &&& enable ability to switch between FAA and GGG, retain last record for other temp instead of making it NaN (see old code)
            except Exception as e: 
                print 'err:',str(e)
                self.state['T_GGG'],self.state['T_FAA'] = nan*units.K, nan*units.K
                self.instruments['Ruox Temperature Monitor'].connected = False
            self.state['datetime'] = datetime.datetime.now()
            self.state['cycle'] += 1
            try: self.state['magnetV'] = yield self.instruments['Magnet Voltage Monitor'].get_magnet_voltage()
            except Exception as e: 
                self.state['magnetV'] = nan*units.V
                self.instruments['Magnet Voltage Monitor'].connected = False
            try:
                self.state['PSCurrent'] = yield self.instruments['Power Supply'].current()
                self.state['PSVoltage'] = yield self.instruments['Power Supply'].voltage()
            except Exception as e:
                self.state['PSCurrent'] = nan*units.A
                self.state['PSVoltage'] = nan*units.V
                self.instruments['Power Supply'].connected = False
            # update relevant files
            try:
                yield self.client.registry.cd(self.ADRSettingsPath)
                file_path = yield self.client.registry.get('Log Path')
            except Exception as e: self.logMessage('Logging temperatures failed.',True)
            with open(file_path+'\\temperatures'+self.dateAppend+'.temps','ab') as f:
                newTemps = [self.state[t]['K'] for t in ['T_60K','T_3K','T_GGG','T_FAA']]
                f.write( struct.pack('d', mpl.dates.date2num(self.state['datetime'])) )
                [f.write(struct.pack('d', temp)) for temp in newTemps]
                #f.write(str(self.state['datetime']) + '\t' + '\t'.join(map(str,newTemps)))
            cycleLength = deltaT(datetime.datetime.now() - cycleStartTime)
            self.client.manager.send_named_message('State Changed', 'state changed')
            #self.stateChanged('state changed')
            yield util.wakeupCall( max(0,self.ADRSettings['step_length']-cycleLength) )
    def _cancelMagUp(self):
        """Cancels the mag up loop."""
        self.state['maggingUp'] = False
        self.logMessage( 'Magging up stopped at a current of '+str(self.state['PSCurrent'])+' Amps.' )
        #self.magUpStopped('cancel') #signal
        self.client.manager.send_named_message('MagUp Stopped', 'cancel')
    @inlineCallbacks
    def _magUp(self):
        """ The magging up method, as per the HPD Manual, involves increasing the voltage in steps of MAG_UP_dV volts
        every cycle of the loop.  This cycle happens once every STEP_LENGTH seconds, nominally 1s (since the voltage
        monitor reads once a second).  Each cycle, the voltage across the magnet is read to get the backEMF.  If it
        is greater than the MAGNET_VOLTAGE_LIMIT, the voltage will not be raised until the next cycle for which the
        backEMF < MAGNET_VOLTAGE_LIMIT. """
        if self.state['maggingUp'] == True:
            self.logMessage('Already magging up.')
            return
        if self.state['regulating'] == True:
            self.logMessage('Currently in PID control loop regulation. Please wait until finished.')
            return
        if self.state['T_3K'] > self.ADRSettings['magnet_max_temp']:
            self.logMessage('Temperature too high to mag up.')
            return
        deviceNames = ['Power Supply','Magnet Voltage Monitor']
        deviceStatus = [self.instruments[name].connected for name in deviceNames]
        if False in deviceStatus:
            message = 'Cannot mag up: At least one of the essential devices is not connected.  Connections: %s'%str([deviceNames[i]+':'+str(deviceStatus[i]) for i in range(len(deviceNames))])
            self.logMessage(message, alert=True)
            return
        self.client.manager.send_named_message('MagUp Started', 'start')
        self.logMessage('Beginning to mag up to '+str(self.ADRSettings['current_limit'])+' Amps.')
        self.state['maggingUp'] = True
        while self.state['maggingUp']:
            startTime = datetime.datetime.now()
            dI = self.state['PSCurrent'] - self.lastState['PSCurrent']
            dt = deltaT( self.state['datetime'] - self.lastState['datetime'] )
            if dt == 0: dt = 0.0000000001 #to prevent divide by zero error
            if self.state['PSCurrent'] < self.ADRSettings['current_limit']:
                if self.state['magnetV'] < self.ADRSettings['magnet_voltage_limit'] and abs(dI/dt) < self.ADRSettings['dIdt_magup_limit'] and self.state['T_FAA'] < self.ADRSettings['magnet_max_temp']:
                    newVoltage = self.state['PSVoltage'] + self.ADRSettings['magup_dV']
                    if newVoltage < self.ADRSettings['voltage_limit']:
                        self.instruments['Power Supply'].voltage(newVoltage) #set new voltage
                    else: self.instruments['Power Supply'].voltage(self.ADRSettings['voltage_limit'])
                    #newCurrent = self.instruments['Power Supply'].current() + 0.005
                    #self.instruments['Power Supply'].current(newCurrent)
                cycleLength = deltaT(datetime.datetime.now() - startTime)
                yield util.wakeupCall( max(0,self.ADRSettings['step_length']-cycleLength) )
            else:
                self.logMessage( 'Finished magging up. '+str(self.state['PSCurrent'])+' Amps reached.' )
                self.state['maggingUp'] = False
                self.client.manager.send_named_message('MagUp Stopped', 'done')
    def _cancelRegulate(self):
        """Cancels the PID regulation loop."""
        self.state['regulating'] = False
        self.logMessage( 'PID Control stopped at a current of '+str(self.state['PSCurrent'])+' Amps.' )
        #self.regulationStopped('cancel')
        self.client.manager.send_named_message('Regulation Stopped', 'cancel')
    @inlineCallbacks
    def _regulate(self,temp): 
        """ This function starts a PID loop to control the temperature.  The basics of it is that a new voltage V+dV is
        proposed.  dV is then limited as necessary, and the new voltage is set. As with magging up, regulate runs a cycle
        at approximately once per second. """
        if self.state['maggingUp'] == True:
            self.logMessage('Currently magging up. Please wait until finished.')
            return
        if self.state['regulating'] == True:
            self.state['regulationTemp'] = temp
            self.logMessage('Setting regulation temperature to %dK.'%temp)
            return
        deviceNames = ['Power Supply','Diode Temperature Monitor','Ruox Temperature Monitor','Magnet Voltage Monitor']
        deviceStatus = [self.instruments[name].connected for name in deviceNames]
        if False in deviceStatus:
            message = 'Cannot regulate: At least one of the essential devices is not connected.  Connections: %s'%str([deviceNames[i]+':'+str(deviceStatus[i]) for i in range(len(deviceNames))])
            self.logMessage(message, alert=True)
            return
        self.client.manager.send_named_message('Regulation Started', 'start')
        self.logMessage( 'Starting regulation to '+str(self.state['regulationTemp'])+'K from '+str(self.state['PSCurrent'])+' Amps.' )
        self.state['regulating'] = True
        print 'beginning regulation'
        print 'V\tbackEMF\tdV/dT\tdV'
        while self.state['regulating']:
            startTime = datetime.datetime.now()
            dI = self.state['PSCurrent'] - self.lastState['PSCurrent']
            if numpy.isnan(self.state['T_FAA']['K']): 
                self.logMessage( 'FAA temp is not valid.  Regulation cannot continue.' )
                self._cancelRegulate()
            print str(self.state['PSVoltage'])+'\t'+str(self.state['magnetV'])+'\t',
            #propose new voltage
            T_target = float(self.state['regulationTemp'])*units.K
            dT = deltaT( self.state['datetime'] - self.lastState['datetime'] )
            #print 'dt =',dT, self.state['datetime'], self.lastState['datetime']
            if dT == 0: dT = 0.0000000001 #to prevent divide by zero error
            self.state['PID_cumulativeError'] += (T_target-self.state['T_FAA'])
            dV = ( self.ADRSettings['PID_KP']*(T_target-self.state['T_FAA']) \
               + self.ADRSettings['PID_KI']*self.state['PID_cumulativeError'] \
               + self.ADRSettings['PID_KD']*(self.lastState['T_FAA']-self.state['T_FAA'])/dT )['K']*units.V
            #hard current limit
            if self.state['PSCurrent'] > self.ADRSettings['current_limit']*units.A:
                if dV>0*units.V: dV=0*units.V
            #hard voltage limit
            if self.state['PSVoltage'] + dV > self.ADRSettings['voltage_limit']*units.V:
                dV = self.ADRSettings['voltage_limit']*units.V - self.state['PSVoltage']
            # steady state limit
            if dV < 0*units.V:
                dV = max(dV,self.state['magnetV']-self.ADRSettings['magnet_voltage_limit']*units.V)
                if dV > 0*units.V: dV = 0*units.V
            if dV > 0*units.V:
                dV = min(dV, self.ADRSettings['magnet_voltage_limit']*units.V-self.state['magnetV'])
                if dV < 0*units.V: dV = 0*units.V
            # limit by hard voltage increase limit
            print str(dV/dT)+'\t',
            if abs(dV/dT) > self.ADRSettings['dVdT_limit']*units.V:
                dV = self.ADRSettings['dVdT_limit']*dT*(dV/abs(dV))*units.V
            # limit by hard current increase limit
            if abs(dI/dT) > self.ADRSettings['dIdt_regulate_limit']*units.A:
                dV = 0*units.V
            # will voltage go negative?
            runCycleAgain = True
            if self.state['PSVoltage']+dV <= 0*units.V:
                self.instruments['Power Supply'].voltage(0*units.V)
                dV = 0*units.V
                runCycleAgain = False
            print str(dV)
            self.instruments['Power Supply'].voltage(self.state['PSVoltage'] + dV)
            cycleTime = deltaT(datetime.datetime.now() - startTime)
            if runCycleAgain: yield util.wakeupCall( max(0,self.ADRSettings['step_length']-cycleTime) )
            else:
                self.logMessage( 'Regulation has completed. Mag up and try again.' )
                self.state['regulating'] = False
                #self.regulationStopped('done') #signal
                self.client.manager.send_named_message('Regulation Stopped', 'done')
    
    @setting(102, 'Get Log', n=['v'], returns=['*(s,b)'])
    def getLog(self,c, n=0):
        """Get an array of the last n logs."""
        if n==0: n = len(self.logMessages)
        n = min(n, len(self.logMessages))
        return [messageAndAlert for messageAndAlert in self.logMessages[-n:]]
    @setting(103, 'Get State Var', var=['s'], returns=['?'])
    def getStateVar(self,c, var):
        """You can get any arbitrary value stored in the state variable by passing its name to this function."""
        return self.state[var]
    @setting(110, 'PSCurrent', returns=['v'])
    def pscurrent(self,c):
        """Get the current of the power supply."""
        return self.state['PSCurrent']
    @setting(111, 'PSVoltage', returns=['v'])
    def psvoltage(self,c):
        """Get the voltage of the power supply."""
        return self.state['PSVoltage']
    @setting(112, 'MagnetV', returns=['v'])
    def magnetv(self,c):
        """Get the voltage across the magnet (at the magnet leads)."""
        #print 'getting magnet voltage',self.state['magnetV']
        return self.state['magnetV']
    @setting(113, 'cycle', returns=['v'])
    def cycle(self,c):
        """How many measurement cycles have been run?"""
        return self.state['cycle']
    @setting(114, 'time', returns=['t'])
    def time(self,c):
        """Returns the time at which the last measurement cycle was run."""
        return self.state['datetime']
    @setting(115, 'Temperatures', returns=['*v'])
    def temperatures(self,c):
        """Returns the measured temperatures in an array: [60K,3K,GGG,FAA]"""
        return [self.state[t] for t in ('T_60K','T_3K','T_GGG','T_FAA')]
    
    @setting(120, 'Regulate', temp=['v'])
    def regulate(self,c, temp=0.1):
        """Starts the PID Temperature control loop."""
        self._regulate(temp)
    @setting(121, 'Mag Up')
    def magUp(self,c):
        """Slowly increases the current through the magnet to the current limit."""
        self._magUp()
    @setting(122, 'Cancel Regulation')
    def cancelRegulation(self,c):
        """Stop PID regulation cycle."""
        self._cancelRegulate()
    @setting(123, 'Cancel Mag Up')
    def cancelMagUp(self,c):
        """Stop mag up process."""
        self._cancelMagUp()
    @setting(124, 'Refresh Instruments')
    def refreshInstruments(self,c):
        """Manually tell all gpib buses to refresh their list of connected devices."""
        self._refreshInstruments()
    @setting(125, 'Add To Log', message=['s'])
    def addToLog(self,c,message=None):
        """Add message to log."""
        if message is not None:
            self.logMessage(message)
    
    @setting(130, 'Set PID KP')
    def setPIDKP(self,c,k=['v']):
        """Set PID Proportional Constant."""
        self.ADRSettings['PID_KP'] = k
        self.logMessage('PID_KP has been set to '+str(k))
    @setting(131, 'Set PID KI')
    def setPIDKI(self,c,k=['v']):
        """Set PID Integral Constant."""
        self.ADRSettings['PID_KI'] = k
        self.logMessage('PID_KI has been set to '+str(k))
    @setting(132, 'Set PID KD')
    def setPIDKD(self,c,k=['v']):
        """Set PID Derivative Constant."""
        self.ADRSettings['PID_KD'] = k
        self.logMessage('PID_KD has been set to '+str(k))


if __name__ == "__main__":
    """Define your instruments here.  This allows for easy exchange between different
    devices to monitor temperature, etc.  For example, the new and old ADR's use two
    different instruments to measure temperature: The SRS module and the Lakeview 218."""
    __server__ = ADRServer(sys.argv)
    util.runServer(__server__)