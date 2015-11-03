import matplotlib as mpl
mpl.use('TkAgg')
import pylab, numpy as np
import Tkinter as tk
import ttk
import tkFileDialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import os, sys
import BNC2110 as bnc 
import threading

# Check out:
# https://pythonhosted.org/PyDAQmx/callback.html

class MeasureIV(tk.Tk):   

    def __init__(self,parent):
        tk.Tk.__init__(self,parent)
        self.parent = parent
        self.running = True
        self.initParams()
        self.initializeWindow()
        self.initializeWaveOutput()
        self.initializeDCOutput()
        self.initializeWaveInput()
        self.after(100,self.measureIV)
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
        self.portOut.set(2)
        self.amp.set(1)
        self.ACFreq.set(1)
        self.ACAmp.set(0.05)
        self.DCAmp.set(0.05)
        self.sampRate.set(10000)
        self.nSamples.set(10000)
        self.portDCIn.trace('w',self.changeDCOutput)
        self.DCAmp.trace('w',self.changeDCOutput)
        self.portOut.trace('w',self.changeWaveInput)
        self.portACIn.trace('w',self.changeWaveOutput)
        self.ACFreq.trace('w',self.changeWaveOutput)
        self.ACAmp.trace('w',self.changeWaveOutput)
        self.sampRate.trace('w',self.changeWaveOutput)
        self.nSamples.trace('w',self.changeWaveOutput)
        self.averages = tk.IntVar()
        self.totalAverages = tk.IntVar()
        self.averages.set(0)
        self.totalAverages.set(1)
        self.averaging = False
        
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
        tk.Entry(fileFrame, width=10, textvariable=self.fileName).pack(side=tk.LEFT)
        tk.Label(fileFrame,text="_#.iv").pack(side=tk.LEFT)
        # (Average and Save||Cancel Averaging) Averages: 0/[#]
        averageFrame = tk.Frame(leftFrame)
        averageFrame.pack(side=tk.TOP)
        self.avgButton = tk.Button(master=averageFrame,text='Average and Save',command=self.averageAndSave)
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
        
        tk.OptionMenu(frame2wire, self.portACIn, 0,1).place(relx=90/597., rely=261/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.ACFreq).place(relx=90/597., rely=285/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.ACAmp).place(relx=90/597., rely=308/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.RACIn).place(relx=305/597., rely=175/578., anchor=tk.CENTER)
        tk.Entry(frame2wire, width=8, textvariable=self.amp).place(relx=450/597., rely=135/578., anchor=tk.CENTER)
        tk.OptionMenu(frame2wire, self.portOut, 0,1,2,3,4,5,6,7).place(relx=531/597., rely=210/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frame3wire, self.portACIn, 0,1).place(relx=93/597., rely=58/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ACFreq).place(relx=93/597., rely=81/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ACAmp).place(relx=93/597., rely=105/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.RACIn).place(relx=144/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ROut).place(relx=405/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.amp).place(relx=411/597., rely=35/578., anchor=tk.CENTER)
        tk.OptionMenu(frame3wire, self.portOut, 0,1,2,3,4,5,6,7).place(relx=545/597., rely=80/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frame4wire, self.portACIn, 0,1).place(relx=58/597., rely=158/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ACFreq).place(relx=58/597., rely=182/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ACAmp).place(relx=58/597., rely=205/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.RACIn).place(relx=38/597., rely=268/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ROut).place(relx=220/597., rely=268/578., anchor=tk.CENTER)
        tk.OptionMenu(frame4wire, self.portOut, 0,1,2,3,4,5,6,7).place(relx=551/597., rely=94/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frameVPhi, self.portDCIn, 0,1).place(relx=94/597., rely=80/578., anchor=tk.CENTER)
        tk.OptionMenu(frameVPhi, self.portACIn, 0,1).place(relx=44/597., rely=192/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ACFreq).place(relx=44/597., rely=216/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ACAmp).place(relx=44/597., rely=239/578., anchor=tk.CENTER)
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
    
    def initializeWaveOutput(self):
        try:
            self.waveOutput = bnc.NIOutputWave(self.portACIn.get(),self.sampRate.get())
            self.waveOutput.setWave(self.genWave( self.ACAmp.get(), self.ACFreq.get() ))
            self.waveOutput.startWave()
            print "started wave output"
        except ValueError:
            pass #invalid value often happens before typing has fully finished
        except Exception as e:
            print 'Error initializing wave output:\n' + str(e)
    
    def initializeDCOutput(self):
        try:
            self.DCOutput = bnc.NIOutputWave(self.portDCIn.get(),self.sampRate.get())
            self.DCOutput.setWave(self.genWave( self.DCAmp.get(), 0 ))
            self.DCOutput.startWave()
            print "started DC output"
        except ValueError:
            pass #invalid value often happens before typing has fully finished
        except Exception as e:
            print 'Error initializing DC output:\n' + str(e)
            
    def initializeWaveInput(self):
        # Create the input task
        self.waveInput = bnc.NIReadWaves2([self.portOut.get()],self.sampRate.get())
        self.waveInput.setCallback(self.updateData)
        self.waveInput.startWave()
        
    def measureIV(self):
        #plot = threading.Thread(target=self.plot_data, args=())
        return
        self.data_read = threading.Thread(target=self.collect_data, args=())
        #might still want the plotting thread since averaging is now slower
        self.data_read.daemon = True
        self.data_read.start()
    
    def updateData(self, data):
        self.VAverages = 0
        self.IAverages = 0
        
        self.cond.acquire()
        try: newdata = data[0]
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
            currents = currents
            voltages = voltages
        elif currentTab == 1: # 3 wire
            currents = currents/self.RACIn.get()/1000
            voltages = voltages/self.amp.get()
        elif currentTab == 2: # 4 wire
            currents = currents/(self.RACIn.get() + self.ROut.get())/1000
            voltages = voltages/self.amp.get()
        elif currentTab == 3: # V-Phi
            currents = currents/self.RACIn.get()/1000
            voltages = voltages/self.amp.get()
            self.ax.set_xlabel('$\Phi$ [A/$\Phi_0$]')
            self.ax.set_ylabel('Voltage [V]')
        
        # average data if selected
        if self.averaging is True and self.averages.get() < self.totalAverages.get():
            self.VAverages = (self.VAverages*self.averages.get() + voltages)/(self.averages.get()+1)
            self.IAverages = (self.IAverages*self.averages.get() + currents)/(self.averages.get()+1)
            self.averages.set(self.averages.get()+1)
            if self.averages.get() == self.totalAverages.get(): # save and re-initialize
                self.saveAveragedData()
                self.cancelAveraging()
        else:
            self.VAverages = voltages
            self.Iaverages = currents
            
        self.plotPoints.set_xdata(self.VAverages)
        self.plotPoints.set_ydata(self.Iaverages)     
        
        self.cond.notify()
        self.cond.release()
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
    
    def collect_data(self):
        self.VAverages = 0
        self.IAverages = 0
        
        while self.running == True:
            self.cond.acquire()
            try: newdata = self.waveInput.readWaves(self.nSamples.get())[0]
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
                currents = currents
                voltages = voltages
            elif currentTab == 1: # 3 wire
                currents = currents/self.RACIn.get()/1000
                voltages = voltages/self.amp.get()
            elif currentTab == 2: # 4 wire
                currents = currents/(self.RACIn.get() + self.ROut.get())/1000
                voltages = voltages/self.amp.get()
            elif currentTab == 3: # V-Phi
                currents = currents/self.RACIn.get()/1000
                voltages = voltages/self.amp.get()
                self.ax.set_xlabel('$\Phi$ [A/$\Phi_0$]')
                self.ax.set_ylabel('Voltage [V]')
            
            # average data if selected
            if self.averaging is True and self.averages.get() < self.totalAverages.get():
                self.VAverages = (self.VAverages*self.averages.get() + voltages)/(self.averages.get()+1)
                self.IAverages = (self.IAverages*self.averages.get() + currents)/(self.averages.get()+1)
                self.averages.set(self.averages.get()+1)
                if self.averages.get() == self.totalAverages.get(): # save and re-initialize
                    self.saveAveragedData()
                    self.cancelAveraging()
            else:
                self.VAverages = voltages
                self.Iaverages = currents
                
            self.plotPoints.set_xdata(self.VAverages)
            self.plotPoints.set_ydata(self.Iaverages)     
            
            self.cond.notify()
            self.cond.release()
            
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
        
        print 'stopping data collection loop'
        self._quit()
    
    def averageAndSave(self):
        self.averaging = True
        self.avgButton.config(text="Cancel Averaging",command=self.cancelAveraging)
    
    def cancelAveraging(self):
        self.averaging = False
        self.averages.set( 0 )
        self.avgButton.config(text='Average and Save',command=self.averageAndSave)
    
    def saveAveragedData(self):
        # &&& add data about n averages, type of measurement, all the resistors and other params
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
        #chooseDirOpts['initialdir'] = expInfo['AFS_Directory']
        chooseDirOpts['mustexist'] = True
        chooseDirOpts['title'] = 'Choose base data directory...'
    	self.savePath.set( tkFileDialog.askdirectory(**chooseDirOpts) )
    
    def changeWaveOutput(self,*args):
        """This should be called (by a listener) every time any of the BNC output port variables change."""
        try: 
            self.waveOutput.endWave()
        except Exception as e: print 'failed to end output wave'
        self.initializeWaveOutput()
    
    def changeDCOutput(self,*args):
        try: 
            self.DCOutput.endWave()
        except Exception as e: print 'failed to end output DC wave', str(e)
        self.initializeDCOutput()
    
    def changeWaveInput(self,*args):
        """This should be called (by a listener) every time any of the BNC input port variables change."""
        try: self.waveInput.endRead()
        except Exception as e: print 'failed to end input wave'
        self.initializeWaveInput()

    def _quit(self):
        """ called when the window is closed."""
        self.running = False
        self.quit()     # stops mainloop
        self.destroy()  # this is necessary on Windows to prevent
                               # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        self.waveOutput.endWave()
        self.waveInput.endRead()
        #os._exit(1)
        
if __name__ == "__main__":
    app = MeasureIV(None)
    app.title("Measure IV")
    app.mainloop()