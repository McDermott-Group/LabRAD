import matplotlib as mpl
mpl.use('TkAgg')
import pylab, numpy as np
import Tkinter as tk
import ttk
import tkFileDialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import os, sys
import niPCI6221 as ni
import threading

# Check out:
# https://pythonhosted.org/PyDAQmx/callback.html

# TODO:
# DC sweep?
# change extension to .pyw to prevent window from opening
# record all parameters used in notes

class MeasureIV(tk.Tk):   

    def __init__(self,parent):
        tk.Tk.__init__(self,parent)
        self.parent = parent
        self.running = True
        self.initParams()
        self.initializeWindow()
        self.initializeACWaves()
        self.initializeDCWave()
        self.lock = threading.Lock()
        self.cond = threading.Condition(threading.Lock())
    
    def initParams(self):
        self.RACIn = tk.DoubleVar()
        self.RDCIn = tk.DoubleVar()
        self.ROut = tk.DoubleVar()
        self.portACIn = tk.IntVar()
        self.portDCIn = tk.IntVar()
        self.portOut = tk.IntVar()
        self.amp = tk.DoubleVar()
        self.ACFreq = tk.DoubleVar()
        self.ACAmp = tk.DoubleVar()
        self.DCAmp = tk.DoubleVar()
        self.sampRate = tk.IntVar()
        self.nSamples = tk.IntVar()
        self.savePath = tk.StringVar()
        self.fileName = tk.StringVar()
        self.RACIn.set(100)
        self.RDCIn.set(100)
        self.ROut.set(100)
        self.portACIn.set(0)
        self.portDCIn.set(1)
        self.portOut.set(0)
        self.amp.set(1)
        self.ACFreq.set(1)
        self.ACAmp.set(0.0)
        self.DCAmp.set(0.0)
        self.sampRate.set(10000)
        self.nSamples.set(10000)
        self.portDCIn.trace('w',self.changeDCOutput)
        self.DCAmp.trace('w',self.changeDCOutput)
        self.portOut.trace('w',self.changeACWaves)
        self.portACIn.trace('w',self.changeACWaves)
        self.ACFreq.trace('w',self.changeACWaves)
        self.ACAmp.trace('w',self.changeACWaves)
        self.sampRate.trace('w',self.changeACWaves)
        self.nSamples.trace('w',self.changeACWaves)
        self.averages = tk.IntVar()
        self.totalAverages = tk.IntVar()
        self.averages.set(0)
        self.totalAverages.set(1)
        self.averaging = False
        self.VAverages = 0
        self.IAverages = 0
        self.savePath.set('.')
        
    def initializeWindow(self):
        """Creates the GUI."""
        root = self
        #set up window
        root.wm_title('Measure IV')
        root.title('Measure IV')
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        #root.geometry("%dx%d+0+0" % (w/2, 0.9*h))
        leftFrame = tk.Frame(root)
        leftFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        rightFrame = tk.Frame(root, width=600)
        rightFrame.pack(side=tk.LEFT)
        
        ### LEFT SIDE ###
        #notes box
        self.comments = tk.Text(master=leftFrame, height=5)
        self.comments.pack(side=tk.TOP, fill=tk.X)
        # IV plot
        self.fig = pylab.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Realtime IV Measurement')
        #   self.ax.set_xlabel('Time')
        #   self.ax.set_ylabel('Temparture [K]')
        self.plotPoints, = self.ax.plot([],[],'-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=leftFrame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #temp plot toolbar at bottom
        self.toolbar = NavigationToolbar2TkAgg( self.canvas, leftFrame )
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # this/is/a/long/file/path/[name].iv
        fileFrame = tk.Frame(leftFrame)
        fileFrame.pack(side=tk.TOP)
        tk.Button(master=fileFrame,text='Select Path',command=self.chooseSaveDirectory).pack(side=tk.LEFT)
        tk.Label(fileFrame,textvariable=self.savePath).pack(side=tk.LEFT)
        tk.Label(fileFrame,text='/').pack(side=tk.LEFT)
        tk.Entry(fileFrame, width=10, textvariable=self.fileName).pack(side=tk.LEFT)
        tk.Label(fileFrame,text="_#.iv").pack(side=tk.LEFT)
        # (Average and Save||Cancel Averaging) Averages: 0/[#]
        averageFrame = tk.Frame(leftFrame)
        averageFrame.pack(side=tk.TOP)
        self.avgButton = tk.Button(master=averageFrame,text='Average',command=self.averageAndSave)
        self.avgButton.pack(side=tk.LEFT)
        tk.Label(averageFrame,text="Averages: ").pack(side=tk.LEFT)
        tk.Label(averageFrame,textvariable=self.averages).pack(side=tk.LEFT)
        tk.Label(averageFrame,text=" / ").pack(side=tk.LEFT)
        tk.Entry(averageFrame, width=8, textvariable=self.totalAverages).pack(side=tk.LEFT)
        #self.fig.tight_layout()
        
        ### RIGHT SIDE ###
        self.measurementTabs = ttk.Notebook(rightFrame, width=600)
        self.measurementTabs.pack(side=tk.TOP)
        frame2wire = ttk.Frame(self.measurementTabs)
        frame3wire = ttk.Frame(self.measurementTabs)
        frame4wire = ttk.Frame(self.measurementTabs)
        frameVPhi  = ttk.Frame(self.measurementTabs)
        self.measurementTabs.add(frame2wire, text='2-Wire')
        self.measurementTabs.add(frame3wire, text='3-Wire')
        self.measurementTabs.add(frame4wire, text='4-Wire')
        self.measurementTabs.add(frameVPhi, text='V-Phi')
        
        bgimg = tk.PhotoImage(file="TwoWire.gif")
        bglabel2 = tk.Label(frame2wire, image=bgimg)
        bglabel2.image = bgimg
        bglabel2.pack()
        bgimg = tk.PhotoImage(file="ThreeWire.gif")
        bglabel3 = tk.Label(frame3wire, image=bgimg)
        bglabel3.image = bgimg
        bglabel3.pack()
        bgimg = tk.PhotoImage(file="FourWire.gif")
        bglabel4 = tk.Label(frame4wire, image=bgimg)
        bglabel4.image = bgimg
        bglabel4.pack()
        bgimg = tk.PhotoImage(file="VPhi.gif")
        bglabelVPhi = tk.Label(frameVPhi, image=bgimg)
        bglabelVPhi.image = bgimg
        bglabelVPhi.pack()
        
        tk.OptionMenu(frame2wire, self.portACIn, 0,1).place(relx=70/597., rely=261/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.ACFreq).place(relx=70/597., rely=285/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.ACAmp).place(relx=70/597., rely=308/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.RACIn).place(relx=305/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.amp).place(relx=450/597., rely=335/578., anchor=tk.CENTER)
        tk.OptionMenu(frame2wire, self.portOut, 0,1,2,3,4,5,6,7).place(relx=531/597., rely=410/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frame3wire, self.portACIn, 0,1).place(relx=86/597., rely=58/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ACFreq).place(relx=86/597., rely=81/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ACAmp).place(relx=86/597., rely=105/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.RACIn).place(relx=144/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ROut).place(relx=405/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.amp).place(relx=411/597., rely=35/578., anchor=tk.CENTER)
        tk.OptionMenu(frame3wire, self.portOut, 0,1,2,3,4,5,6,7).place(relx=545/597., rely=80/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frame4wire, self.portACIn, 0,1).place(relx=41/597., rely=158/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ACFreq).place(relx=41/597., rely=182/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ACAmp).place(relx=41/597., rely=205/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.RACIn).place(relx=38/597., rely=268/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ROut).place(relx=220/597., rely=268/578., anchor=tk.CENTER)
        tk.OptionMenu(frame4wire, self.portOut, 0,1,2,3,4,5,6,7).place(relx=551/597., rely=94/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frameVPhi, self.portDCIn, 0,1).place(relx=94/597., rely=80/578., anchor=tk.CENTER)
        tk.OptionMenu(frameVPhi, self.portACIn, 0,1).place(relx=34/597., rely=194/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ACFreq).place(relx=34/597., rely=218/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ACAmp).place(relx=34/597., rely=241/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.DCAmp).place(relx=94/597., rely=105/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.RDCIn).place(relx=144/597., rely=156/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.RACIn).place(relx=94/597., rely=306/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ROut).place(relx=405/597., rely=156/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.amp).place(relx=411/597., rely=36/578., anchor=tk.CENTER)
        tk.OptionMenu(frameVPhi, self.portOut, 0,1,2,3,4,5,6,7).place(relx=545/597., rely=80/578., anchor=tk.CENTER)
        
        self.measurementTabs.select(1)
        self.currentTab = 1
        self.measurementTabs.bind_all("<<NotebookTabChanged>>", self.tabChangedEvent)
        
        root.protocol("WM_DELETE_WINDOW", self._quit) #X BUTTON

    def tabChangedEvent(self, event):
        tabid = self.measurementTabs.select()
        self.currentTab = self.measurementTabs.index(tabid)
    
    def genWave(self, amp, freq):
        """
        Creates an output wave vector.
        Returns wave in np.float64 array.
        """
        # Setting freq to 0 is just a DC output.
        # Number of samples doesn't really matter in
        #   that case, so just set to the sample rate.
        if freq is not 0:
            samps = int(float(self.sampRate.get()) / freq)
        else:
            samps = int(self.sampRate.get())
        
        # Generate empty wave of correct size
        wave = np.zeros((samps,),dtype=np.float64)
        
        # Sample the wave at sampRate. Use cos such
        #   that the case of freq=0 will return DC amp.
        for n in range(samps):
            wave[n] = amp * np.cos(2*np.pi*n*freq/samps)
        
        # Return the wave to the caller
        return wave
    
    def initializeACWaves(self):
        try:
            writeBuf = self.genWave(self.ACAmp.get(), self.ACFreq.get())
            
            self.waveInput = ni.CallbackTask()
            self.waveInput.configureCallbackTask("Dev1/ai"+str(self.portOut.get()), self.sampRate.get(),len(writeBuf))
            self.waveInput.setCallback(self.updateData)
            triggerName = self.waveInput.getTrigName()
            
            self.waveOutput = ni.acAnalogOutputTask()
            self.waveOutput.configureAcAnalogOutputTask("Dev1/ao"+str(self.portACIn.get()), self.sampRate.get(),writeBuf,trigName=triggerName)
            
            self.waveOutput.StartTask()
            self.waveInput.StartTask()
            print "started AC waves"
        except ValueError:
            pass #invalid value often happens before typing has fully finished
        except Exception as e:
            print 'Error initializing wave output:\n' + str(e)
    
    def initializeDCWave(self):
        try:
            self.DCOutput = ni.dcAnalogOutputTask()
            self.DCOutput.configureDcAnalogOutputTask("Dev1/ao"+str(self.portDCIn.get()),self.DCAmp.get())
            self.DCOutput.StartTask()
            print "started DC output"
        except ValueError:
            pass #invalid value often happens before typing has fully finished
        except Exception as e:
            print 'Error initializing DC output:\n' + str(e)
        
    def updateData(self, data):
        try: self.ACAmp.get(), self.ACFreq.get()
        except ValueError: return
        
        self.cond.acquire()
        
        try: newdata = data
        except Exception as e:
            print 'failed to aquire data'
            newdata = []
        #this is dummy data. Uncomment the line above
        #newdata = [10 * np.random.random_sample(10),10* np.random.random_sample(10)]
        
        currents = np.array(self.genWave(self.ACAmp.get(),self.ACFreq.get()))
        voltages = np.array(newdata)
        
        #print 'tab id',self.measurementTabs.select(), self.measurementTabs.index(self.measurementTabs.select())
        #tabid = self.measurementTabs.select()
        currentTab = self.currentTab#measurementTabs.index(tabid)
        self.ax.set_xlabel('Voltage [V]')
        self.ax.set_ylabel('Current [A]')
        if currentTab == 0: # 2 wire
            try: 
                currents = (currents-voltages/self.amp.get())/self.RACIn.get()
                voltages = voltages/self.amp.get()
                currents, voltages = voltages, currents
                self.ax.set_xlabel('Current [A]')
                self.ax.set_ylabel('Voltage [V]')
            except ValueError: pass # in case the fields have bad values or are not finished typing
        elif currentTab == 1: # 3 wire
            try:
                currents = currents/self.RACIn.get()/1000
                voltages = voltages/self.amp.get()
            except ValueError: pass
        elif currentTab == 2: # 4 wire
            try:
                currents = currents/(self.RACIn.get() + self.ROut.get())/1000
                voltages = voltages/self.amp.get()
            except ValueError: pass
        elif currentTab == 3: # V-Phi
            try:
                currents = currents/self.RACIn.get()/1000
                voltages = voltages/self.amp.get()
                currents, voltages = voltages, currents
                self.ax.set_xlabel('$\Phi/L$ [A]')
                self.ax.set_ylabel('Voltage [V]')
            except ValueError: pass
        
        # average data if selected
        if self.averaging is True and self.averages.get() < self.totalAverages.get():
            self.VAverages = (self.VAverages*self.averages.get() + voltages)/(self.averages.get()+1.)
            self.IAverages = (self.IAverages*self.averages.get() + currents)/(self.averages.get()+1.)
            self.averages.set(self.averages.get()+1)
            if self.averages.get() == self.totalAverages.get(): # save and re-initialize
                self.saveAveragedData()
                self.cancelAveraging()
        else:
            self.VAverages = voltages
            self.IAverages = currents
            
        self.plotPoints.set_xdata(self.VAverages)
        self.plotPoints.set_ydata(self.IAverages)     
        
        self.cond.notify()
        self.cond.release()
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
    
    def averageAndSave(self):
        self.averaging = True
        self.avgButton.config(text="Cancel Averaging",command=self.cancelAveraging)
    
    def cancelAveraging(self):
        self.averaging = False
        self.averages.set( 0 )
        if self.savePath.get() != '.': btntext = 'Average and Save'
        else: btntext = 'Average'
        self.avgButton.config(text=btntext,command=self.averageAndSave)
    
    def saveAveragedData(self):
        # &&& add data about n averages, type of measurement, all the resistors and other params
        if self.savePath.get() != '.':
            i = 1
            while True:
                fullSavePath = os.path.join(self.savePath.get(),(self.fileName.get()+'_%03d.iv'%i))
                if not os.path.exists(fullSavePath): break
                i += 1
            with open(fullSavePath,'a') as f:
                dataToSave = np.transpose(np.asarray([self.IAverages,self.VAverages]))
                f.write(self.comments.get(1.0, tk.END))
                np.savetxt(f,dataToSave)
    
    def chooseSaveDirectory(self):
    	chooseDirOpts = {}
        currentTab = self.currentTab
        if currentTab == 0:
            chooseDirOpts['initialdir'] = 'Z:\\mcdermott-group\\Data\\Suttle Data\\Nb\\'
        #chooseDirOpts['initialdir'] = expInfo['AFS_Directory']
        chooseDirOpts['mustexist'] = True
        chooseDirOpts['title'] = 'Choose base data directory...'
    	self.savePath.set( tkFileDialog.askdirectory(**chooseDirOpts) )
        self.avgButton.config(text="Average and Save")
    
    def changeACWaves(self,*args):
        """This should be called (by a listener) every time any of the BNC output port variables change."""
        try:
            self.waveOutput.StopTask()
            self.waveOutput.ClearTask()
            self.waveInput.StopTask()
            self.waveInput.ClearTask()
        except: print 'failed to end wave'
        # if port is changed, we should automatically switch AC and DC ports
        if self.portACIn.get() == self.portDCIn.get():
            self.portDCIn.set((self.portACIn.get()+1)%2)
        try: 
            self.ACAmp.get(), self.ACFreq.get() #raise error if cell is not valid float
            self.initializeACWaves()
        except ValueError: pass # if cell is not valid float
        except Exception as e: print 'failed to start wave', str(e)
    
    def changeDCOutput(self,*args):
        try:
            self.DCOutput.StopTask()
            self.DCOutput.ClearTask()
        except: print 'failed to end DC wave'
        # if port is changed, we should automatically switch AC and DC ports
        if self.portACIn.get() == self.portDCIn.get():
            self.portACIn.set((self.portDCIn.get()+1)%2)
        try: 
            self.DCAmp.get()    # raise error if cell is not valid float
            self.initializeDCWave()
        except ValueError: pass # if cell is not valid float
        except Exception as e: print 'failed to start DC wave', str(e)

    def _quit(self):
        """ called when the window is closed."""
        self.ACAmp.set(0)
        self.DCAmp.set(0)
        self.running = False
        self.quit()     # stops mainloop
        self.destroy()  # this is necessary on Windows to prevent
                               # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        self.waveOutput.StopTask()
        self.waveOutput.ClearTask()
        self.waveInput.StopTask()
        self.waveInput.ClearTask()
        self.DCOutput.StopTask()
        self.DCOutput.ClearTask()
        #os._exit(1)
        
if __name__ == "__main__":
    app = MeasureIV(None)
    app.title("Measure IV")
    app.mainloop()