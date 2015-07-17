"""
### BEGIN NODE INFO
[info]
name = ADR Controller GUI
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

class QPCDC(object):#Tkinter.Tk):
    """Provides a GUI to measure the DC steps on a QPC"""
    name = 'QPC DC Measurments'
    ID = 6117
    
    def __init__(self,parent,peripheralDict):
        #Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.periphs = peripheralDict
        #initialize and start measurement loop
        self.connect()
        self.initializeWindow()
    @inlineCallbacks
    def connect(self,cxn=None):
        """Connects to labrad, loads the last 20 log messages, and starts listening for messages from the adr server."""
        if cxn == None:
            #make an asynchronous connection to LabRAD
            from labrad.wrappers import connectAsync
            self.cxn = yield connectAsync(name = self.name)
        else:self.cxn = cxn
        self.voltageSource = self.cxn[ self.periphs['Voltage Source'][0] ]
        yield self.voltageSource.select_device( self.periphs['Voltage Source'][1] )
        
    def initializeWindow(self):
        """Creates the GUI."""
        root = self.parent
        #set up window
        root.wm_title('QPC DC Measurement')
        root.title('QPC DC Measurement')
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry("%dx%d+0+0" % (w/2, 0.9*h))
        #comment box log
        self.log = Tkinter.Text(master=root, height=5)
        self.log.pack(side=Tkinter.TOP, fill=Tkinter.X)
        # temp plot
        self.fig = pylab.figure()
        self.ax = self.fig.add_subplot(111)
        #self.ax2 = self.ax.twinx()
        self.ax.set_title('Realtime QPC Measurement')
        self.ax.set_xlabel('$V_{gate}$ [V]')
        self.ax.set_ylabel('Output Voltage [V]')
        self.graph, = self.ax.plot([],[])
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        #temp plot toolbar at bottom
        self.toolbar = NavigationToolbar2TkAgg( self.canvas, root )
        self.toolbar.update()
        #self.toolbar.pack(side=Tkinter.BOTTOM, fill=Tkinter.X)
        self.canvas._tkcanvas.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        
        #shows settings
        settingsFrame = Tkinter.Frame(root)
        settingsFrame.pack(side=Tkinter.TOP)
        self.startV = Tkinter.DoubleVar()
        self.stopV = Tkinter.DoubleVar()
        self.nPoints = Tkinter.IntVar()
        Tkinter.Label(settingsFrame, text="Range: -").pack(side=Tkinter.LEFT)
        Tkinter.Entry(settingsFrame, textvariable=self.startV).pack(side=Tkinter.LEFT)
        Tkinter.Label(settingsFrame, text=" to -").pack(side=Tkinter.LEFT)
        Tkinter.Entry(settingsFrame, textvariable=self.stopV).pack(side=Tkinter.LEFT)
        Tkinter.Label(settingsFrame, text="V with ").pack(side=Tkinter.LEFT)
        Tkinter.Entry(settingsFrame, textvariable=self.nPoints).pack(side=Tkinter.LEFT)
        Tkinter.Label(settingsFrame, text="points").pack(side=Tkinter.LEFT)
        self.startDataTaking = Tkinter.Button(settingsFrame, text='Go!', command=self.takeData)
        self.startDataTaking.pack(side=Tkinter.LEFT)
        self.saveButton = Tkinter.Button(settingsFrame, text='Save', command=self.save)
        self.saveButton.pack(side=Tkinter.LEFT)
        
        self.fig.tight_layout()
        
        root.protocol("WM_DELETE_WINDOW", self._quit) #X BUTTON
        
    def stopTakingData(self):
        self.takingData = False
        self.startDataTaking.configure(text='Go!', command=self.takeData)
    @inlineCallbacks
    def takeData(self):
        self.startDataTaking.configure(text='Stop', command=self.stopTakingData)
        self.takingData = True
        start = -1*self.startV.get()
        stop = -1*self.stopV.get()
        print 'taking data', start, stop
        for v in numpy.arange(start,stop,-(self.stopV.get()-self.startV.get())/max(self.nPoints.get(),1)):
            if self.takingData == False: break
            self.voltageSource.voltage(v)
            Vout = v#&&&
            self.graph.set_xdata(numpy.append(self.graph.get_xdata(),v))
            self.graph.set_ydata(numpy.append(self.graph.get_ydata(),Vout))
            if self.toolbar._active == 'HOME' or self.toolbar._active == None:
                self.ax.set_xlim(stop,start)
                self.ax.relim()
            self.ax.autoscale(axis='y')
            #draw
            self.canvas.draw()
            yield util.wakeupCall( 1 )
        self.startDataTaking.configure(text='Go!', command=self.takeData)
        self.takingData = False
        print 'finished'
    def save(self):
    	with open(FILEPATH,'w') as f:
			saveText = self.log.get(1.0,Tkinter.END)
			saveText += dt.strftime('\ndata taking started: %m/%d/%y %H:%M:%S\n')
			f.write(saveText)
			for i in range(len(self.graph.get_xdata())):
    			f.write(str(self.graph.get_xdata()[i])+'\t'+str(self.graph.get_ydata()[i]))
    def _quit(self):
        """ called when the window is closed."""
        self.parent.quit()     # stops mainloop
        self.parent.destroy()  # this is necessary on Windows to prevent
                               # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        reactor.stop()
        
if __name__ == "__main__":
    peripheralDict = {  'Voltage Source':['SIM928 Server','SIM900 SRS Mainframe - GPIB0::2::SIM900::4'] } #{'device',['name','addr']}
    mstr = Tkinter.Tk()
    tksupport.install(mstr)
    app = QPCDC(mstr,peripheralDict)
    reactor.run()