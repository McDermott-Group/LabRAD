"""
### BEGIN NODE INFO
[info]
name = 
version = 1.3.2-no-refresh
description = This is a simple labrad client that gives a GUI interface to ADRServer, which controls our ADRs

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 20
### END NODE INFO
"""

import matplotlib as mpl
mpl.use('TkAgg')
import pylab, numpy
import datetime
import Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import labrad
from labrad.server import (inlineCallbacks, returnValue)
from twisted.internet import tksupport, reactor
# from labrad.units import Unit,Value
from labrad import util

#FILEPATH = 'Z:\physics\mcdermott-group\cwilen\BN JJ (Purdue)\R Data 20151218\\resC2.txt'
FILEPATH = 'C:\Users\Robert McDermott\Desktop\R Data 20160203\\resDown.txt'
CYCLE_TIME = 4 # in seconds
TEMPCHAN = 6

import niPCI6221 as ni
OUTPUT_V = 0.5
BN_PORT = 0
NbSe2_PORT = 1
AVERAGES = 20

class ResCal(object):
    """Provides a GUI to measure the DC steps on a QPC"""
    name = 'Resistance Calibration'
    ID = 6117
    
    def __init__(self,parent,peripheralDict):
        self.parent = parent
        self.periphs = peripheralDict
        #initialize and start measurement loop
        self.connect()
        self.initializeWindow()
        self.initializeMeasurement()
    @inlineCallbacks
    def connect(self,cxn=None):
        """Connects to labrad, loads the last 20 log messages, and starts listening for messages from the adr server."""
        if cxn == None:
            #make an asynchronous connection to LabRAD
            from labrad.wrappers import connectAsync
            self.cxn = yield connectAsync(name = self.name)
        else:self.cxn = cxn
        #self.DMM = self.cxn[ self.periphs['DMM'][0] ]
        #yield self.DMM.select_device( self.periphs['DMM'][1] )
        self.diodeMonitor = self.cxn[ self.periphs['Diode Monitor'][0] ]
        yield self.diodeMonitor.select_device( self.periphs['Diode Monitor'][1] )
        
    def initializeWindow(self):
        """Creates the GUI."""
        root = self.parent
        #set up window
        root.wm_title('Resistor Calibration')
        root.title('Resistor Calibration')
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry("%dx%d+0+0" % (w/2, 0.9*h))
        #comment box log
        self.log = Tkinter.Text(master=root, height=5)
        self.log.pack(side=Tkinter.TOP, fill=Tkinter.X)
        # temp plot
        self.fig = pylab.figure()
        self.ax = self.fig.add_subplot(111)
        #self.ax2 = self.ax.twinx()
        self.ax.set_xlabel('Temperature [K]')
        self.ax.set_ylabel('Resistance [$\Omega$]')
        self.graph, = self.ax.plot([],[],'.')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        #temp plot toolbar at bottom
        self.toolbar = NavigationToolbar2TkAgg( self.canvas, root )
        self.toolbar.update()
        #self.toolbar.pack(side=Tkinter.BOTTOM, fill=Tkinter.X)
        self.canvas._tkcanvas.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        
        #shows settings
        self.startStopButton = Tkinter.Button(root, text='Start', command=self.takeData)
        self.startStopButton.pack(side=Tkinter.LEFT)
        
        self.fig.tight_layout()
        
        root.protocol("WM_DELETE_WINDOW", self._quit) #X BUTTON
        
    def initializeMeasurement(self):
        try:
            self.DCOutput_BN = ni.dcAnalogOutputTask()
            self.DCOutput_BN.configureDcAnalogOutputTask("Dev1/ao"+str(BN_PORT),OUTPUT_V)
            self.DCOutput_BN.StartTask()
            self.DCOutput_NbSe2 = ni.dcAnalogOutputTask()
            self.DCOutput_NbSe2.configureDcAnalogOutputTask("Dev1/ao"+str(NbSe2_PORT),OUTPUT_V)
            self.DCOutput_NbSe2.StartTask()
            print "started DC outputs"
        except Exception as e:
            print 'Error initializing DC outputs:\n' + str(e)
    def stopTakingData(self):
        self.takingData = False
        self.startStopButton.configure(text='Start', command=self.takeData)
    @inlineCallbacks
    def takeData(self):
        self.startStopButton.configure(text='Stop', command=self.stopTakingData)
        self.takingData = True
        print 'taking data'
    	with open(FILEPATH,'a') as f:
			saveText = self.log.get(1.0,Tkinter.END)
			dt = datetime.datetime.now()
			saveText += dt.strftime('\ndata taking started: %m/%d/%y %H:%M:%S\n')
			f.write(saveText)
        while self.takingData:
            #Ts = yield self.diodeMonitor.get_temperature()
            #T = Ts[TEMPCHAN]
            T = yield self.cxn.adr2.temperatures()
            T = T[1]
            #R = yield self.DMM.get_resistance()
            #R = yield self.DMM.get_measurement('4-wire ohms','3Mohms')
            #R = yield self.DMM.get_fw_resistance()
            t_BN = ni.analogInputTask("Dev1/ai"+str(BN_PORT),AVERAGES,AVERAGES)
            V_BN = numpy.mean(t_BN.configureCallbackTask())
            t_NbSe2 = ni.analogInputTask("Dev1/ai"+str(NbSe2_PORT),AVERAGES,AVERAGES)
            V_NbSe2 = numpy.mean(t_NbSe2.configureCallbackTask())
            R_BN = OUTPUT_V*10000/(V_BN) - 10000
            R_NbSe2 = OUTPUT_V*10000/(-V_NbSe2) - 10000
            # print T,R
            with open(FILEPATH,'a') as f:
                f.write(str(T)+'\t'+str(R_BN)+'\t'+str(R_NbSe2)+'\n')
            self.graph.set_xdata(numpy.append(self.graph.get_xdata(),T))
            self.graph.set_ydata(numpy.append(self.graph.get_ydata(),R_BN))
            self.ax.relim()
            self.ax.autoscale()
            self.canvas.draw()
            yield util.wakeupCall( CYCLE_TIME )
        self.startStopButton.configure(text='Start', command=self.takeData)
        self.takingData = False
        print 'finished'
    def _quit(self):
        """ called when the window is closed."""
        self.parent.quit()     # stops mainloop
        self.parent.destroy()  # this is necessary on Windows to prevent
                               # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        reactor.stop()
        
if __name__ == "__main__":
    peripheralDict = {  'Diode Monitor':['Lakeshore 218','mcd-adr1 GPIB Bus - GPIB0::18::INSTR'],'DMM':['Keithley 2000 DMM','mcd-adr1 GPIB Bus - GPIB0::16::INSTR']}#['Keithley 2000 DMM','mcd-adr1 GPIB Bus - GPIB0::16::INSTR'] } #{'device',['name','addr']}
    mstr = Tkinter.Tk()
    tksupport.install(mstr)
    app = ResCal(mstr,peripheralDict)
    reactor.run()