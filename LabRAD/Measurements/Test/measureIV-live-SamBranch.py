"""
---------------------------
PROGRAM INFO
measureIV.py program for using a National Instrument BNC-2110 to generate
    an output waveform to a device and measure the device response, plot it
    in real-time for the user, perform averaging as requested, and ultimately
    save data in a defined directory-tree.
    
Author: Ed Leonard
Version Date: 2015-Mar-20

---------------------------

---------------------------
SYSTEM REQUIREMENTS
    * NI-DAQmx. This is able to be installed without LabVIEW from
        https://software.wisc.edu/cgi-bin/ssl/csl.cgi via downloading the LabVIEW
        driver installation package (but not the entire LabVIEW system!).
        
        ** When installing NI-DAQmx, ensure that the "ANSI C Support" option is 
            installed or this will not work.
    
    * PyDAQmx. This is a Python wrapper for NI-DAQmx's ANSI C capabilities. 
        It's available here: http://pythonhosted.org/PyDAQmx/
    
    * All other packages required are standard within the scipy stack. Most
        testing was done using the Anaconda package for Python 2.7 x64 available 
        here: http://continuum.io/downloads
        
---------------------------

---------------------------
INSTRUCTIONS
    X.) If issues arise, please email emleonard@wisc.edu
    1.) Start Program
    2.) Define channels for input/output on the BNC device.
    3.) Choose output save directory and enter experimental information.
    4.) Adjust output wave/averaging to acquire desired data.
    5.) Save data.
    6.) Repeat as necessary.

---------------------------    

---------------------------
KNOWN ISSUES
    * averaging doesn't work
    * resetting the average doesn't work
    * need ability to flip axes easily
    * saving data doesn't work
    * changing the frequency and amplitude doesn't yet work (add button?)
    * doesn't actually take data in yet

---------------------------    

"""

# Default application parameters initially defined in experimental info dictionary.

expInfo = { 'AFS_Directory':        'C:\\',
            'deviceType':           '',
            'waferName':            '',
            'dieName':              '',
            'measurementType':      'IV',
            'measurementLocation':  '',
            'nSamps':               10000,
            'sampRate':             10000,
            'comments':             '',
            'nAvgs':                1,
            'currentAverages':      1,
            'outputWaveAmp':        0,
            'outputWaveFreq':       1,       
            'bncXin':               0,
            'bncYin':               1,
            'bncVout':              0   }

# System packages
import datetime
from time import strftime
import os

# scipy packages
from pylab import *
import numpy as np
from matplotlib.figure import Figure

# GUI packages
import Tkinter as tk
import ttk
import tkFileDialog
import tkMessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# NI BNC-2110 classes package
import BNC2110 as bnc 

# Threading modules
import threading
from sys import exit
from time import sleep

# Max values for BNC-2110 tool
NI_BNC_MAX_RATE = 10000 # Hz
NI_BNC_MAX_VOLT = 10 # volts

# Runtime variables
REFRESH_INTERVAL = 0 # ms
PLOT_REFRESH_INTERVAL = 0 # ms


###################################################
"""
Function that uses expInfo dictionary to create
    an output wave vector.

Zero inputs required (expInfo is implied global)

Function returns wave in np.float64 array.

"""
###################################################
def genWave():
    
    # Grab the most current information
    amp = float(expInfo['outputWaveAmp'])
    freq = float(expInfo['outputWaveFreq'])
    
    # Setting freq to 0 is just a DC output.
    # Number of samples doesn't really matter in
    #   that case, so just set to the sample rate.
    if freq is not 0:
        samps = int(expInfo['sampRate'] / freq)
    else:
        samps = int(expInfo['sampRate'])
    
    # Generate empty wave of correct size
    wave = np.zeros((samps,),dtype=np.float64)
    
    # Sample the wave at sampRate. Use cos such
    #   that the case of freq=0 will return DC amp.
    for n in range(samps):
        wave[n] = amp * np.cos(2*np.pi*n*freq/samps)

    # Return the wave to the caller
    return wave

###################################################
"""
Function that uses expInfo dictionary to create
    string for output file path/name.

Zero inputs required (expInfo is implied global)

Function returns nothing, but places resulting
    file path within expInfo.

"""
###################################################
def setDataDir():
    
    # Get today's date
    now = datetime.datetime.now()
    expInfo['today'] = strftime("%d-%m-%Y")
    
    # Generate file path.
    expInfo['saveDir'] = os.path.join(expInfo['AFS_Directory'],
                           expInfo['deviceType'],
                           expInfo['waferName'],
                           expInfo['dieName'],
                           expInfo['measurementType'],
                           expInfo['today'])
   
    # Propose a filename and check its existence. If it doesn't exist, create and open it.
    #     If it does exist, increment counter and try again.
    fileExists = True
    n = 0
    while fileExists == True:
        expInfo['saveFileNum'] = "%03d"%n
        expInfo['saveFileName'] = (     expInfo['measurementType'] + "-" +  
                                        expInfo['today'] + "_" + 
                                        expInfo['saveFileNum'] +  ".txt" )
        expInfo['saveFile'] = os.path.join(expInfo['saveDir'],expInfo['saveFileName'])
        
        fileExists = os.path.isfile(expInfo['saveFile'])
        n += 1
    expInfo['saveFile'] = os.path.normpath(expInfo['saveFile'])
    #print expInfo['saveFile']
    
    #f = open(expInfo['saveFile'],'w+')
    
    return

###################################################
""" 
INCOMPLETE FUNCTION

Input data array required

Writes data to file in neat fashion yet to be set.

"""
###################################################
def writeDataToFile(data_str):
    setDataDir()
    
    if not os.path.exists(expInfo['saveDir']):
        try:
            os.makedirs(expInfo['saveDir'])
        except:
            print "Error in making directory tree. Check path permissions/AFS Credentials"
            return False
            
            
    f = open(expInfo['saveFile'],'w+')
    f.write(data_str+"\n")
    f.close()
    return True


###################################################
""" 
INCOMPLETE CLASS

No inputs required.

Class contains application window management with
    functions required for data acquisition and 
    output wave generation via the BNC2110 classes. 
    
"""
###################################################

class MeasureIV(tk.Tk):   

    def __init__(self,parent):
        tk.Tk.__init__(self,parent)
        self.parent = parent
        self.saveDirChosen = False
        self.bncMarkedUpdate = False
        self.updatingPlot = False
        self.initialize() 
        self.iterations = 0
        self.running = True
        self.oldConf = {}
        
        self.inputPars = ['bncXin','bncYin','sampRate']


    def updateWindow(self):
        
        self.oldWave = [expInfo['outputWaveAmp'],expInfo['outputWaveFreq']]
        
        for key in self.enterz:
            expInfo[key] = self.enterz[key].get()
        
        if self.saveDirChosen == True:
            setDataDir()
            self.directoryLabel.config(text="Save Path: " + expInfo['saveFile'],
                                       anchor="w",relief="groove",fg = "black")
        # Check if the BNC settings have changed since last update loop
        #     If they have, kill old handles and open new ones.
        if self.bncMarkedUpdate == True:
            self.bncMarkedUpdate = False
            
            if self.oldConf['bncVout'] is not expInfo['bncVout'] or self.oldConf['sampRate'] is not expInfo['sampRate']:
                self.waveOutput.endWave()
                self.initializeWaveOutput()
            
            if self.oldConf['bncXin'] is not expInfo['bncXin'] or self.oldConf['bncYin'] is not self.expInfo['bncYin'] or self.oldConf['sampRate'] is not expInfo['sampRate']:
                self.waveInput.endRead()
                self.initializeWaveInput()
            
        # Check if a new wave needs to be written to the BNC device.
        if self.oldWave[0] is not expInfo['outputWaveAmp'] or self.oldWave[1] is not expInfo['outputWaveFreq']:

            # If it does, reinitialize it.
            self.waveOutput.endWave()
            self.initializeWaveOutput()
        
        # Initialize plot loop and re-run the updateWindow method every REFRESH_INTERVAL ms.
        if self.updatingPlot == False:
            self.updatePlot()
        
        self.after(REFRESH_INTERVAL,self.updateWindow)
    
    def updatePlot(self):
        self.updatingPlot = True
        
        plot = threading.Thread(target=self.plot_data, args=())
        data_read = threading.Thread(target=self.collect_data, args=())
        
        # This is sample data that updates once per function call just to 
        #     make sure that the graph updates properly.
        new_x = arange(0,3,0.01)
        new_y = np.sin(2*np.pi)
        # self.nums += 0.2
        
        # If it hasn't already, create a figure and start reading
        # from the BNC brick.
        if not hasattr(self,'fig'):
            self.fig = Figure(dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title('Literally the Title')
            self.ax.grid()
            
            self.pltcanvas = FigureCanvasTkAgg(self.fig,master=self)
            self.pltcanvas.show()
            self.pltcanvas.get_tk_widget().grid(row=8,column=0,rowspan=1,columnspan=4)
            
            self.line = self.ax.plot([],[],'r-')[0]
            
            # Start daq task for the input waves.
            self.initializeWaveInput()
            
            plot.start()
            data_read.start()
        '''
        collect = threading.Thread(target=self.collect_data, args=())
        
        def worker():
            while True:
                item = q.get()
                do_work(item)
                q.task_done()

        q = Queue()
        for i in range(num_worker_threads):
             t = Thread(target=worker)
             t.daemon = True
             t.start()

        for item in source():
            q.put(item)

        q.join()
        
        '''    
        
        # Continue updating as often as possible.
        self.after(PLOT_REFRESH_INTERVAL,self.updatePlot)
        if not self.running:            
            os._exit(1)
        
        
    def collect_data(self):
        sleep(1)
        while (self.running):
            print "got data"
            self.newdata = self.getNewData()
            self.line.set_xdata(self.newdata[0])
            self.line.set_ydata(self.newdata[1])
        return
            
    def plot_data(self):
        while (self.running):
            print "drawing data"
            sleep(5)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw
        return
        '''
        print self.ax
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw
        return
        '''
        
    def startAverage(self):
        # This is where data averaging must be handled.
        return
    
    def resetAverage(self):
        # A running average is now reset and starts over.
        
        
        self.startAverage
        return
    
    def saveData(self):
        
        for key in self.enterz:
            expInfo[key] = self.enterz[key].get()
        data = "What up?"
        self.data_written = writeDataToFile(data)
        if self.data_written == False:
            tkMessageBox.showinfo("Save Error","An error has occurred in saving.\n Please check permissions and AFS credentials.")
            
    
    def askDirectory(self):
        # Pops up a file dialog for selecting a base data-saving directory.
        expInfo['AFS_Directory'] = tkFileDialog.askdirectory(**self.chooseDirOpts)
        
        # Used as a check elsewhere as to whether or not the user is prepared
        #     to save data or not.
        self.saveDirChosen = True
        self.buttonz['save'].config(state="active",command=self.saveData)
        
    def initialize(self):
        
        # This method writes the GUI for display. It is only called once at program startup.
        
        # self.headerLabel = tk.Label(self, text="Measure IV")
        self.directoryLabel = tk.Label(self, text="PLEASE CHOOSE DATA DIRECTORY",relief="groove",fg="red")
        
        # self.headerLabel.grid(row = 0, column = 0, columnspan = 6)
        self.directoryLabel.grid(row=12,column=0,columnspan=4)
        
        # Define a dictionary filled with [tk.Labels, row_num, col_num] to generate
        #    labels and entry fields usable by the user.
        self.labelz = {}
        self.labelz['deviceType'] = [tk.Label(self,text="Device Type:"),1,2]
        self.labelz['measurementType'] = [tk.Label(self,text="Measurement Type:"),1,0]
        self.labelz['waferName'] = [tk.Label(self,text="Wafer Name:"),2,0]
        self.labelz['dieName'] = [tk.Label(self,text="Die Name:"),2,2]
        self.labelz['measurementLocation'] = [tk.Label(self,text="Measurement Location:"),3,0]
        # self.labelz['sampRate'] = [tk.Label(self,text="Sampling Rate (Hz):"),4,0]
        # self.labelz['nSamps'] = [tk.Label(self,text="Number of Samples:"),4,2]
        self.labelz['comments'] = [tk.Label(self,text="Comments:"),3,2]
        self.labelz['nAvgs'] = [tk.Label(self,text="Averages:"),9,0]
        self.labelz['currentAverages'] = [tk.Label(self,text="Current Averages:"),9,2]
        self.labelz['outputWaveAmp'] = [tk.Label(self,text="Output Wave Amplitude (V):"),5,0]
        self.labelz['outputWaveFreq'] = [tk.Label(self,text="Output Wave Frequency (Hz):"),5,2]
        
        self.enterz = {}
        
        # Loop over all labelz and create each with an associated entry field.
        for key in self.labelz:
            self.enterz[key] = tk.Entry(self,width="13")
            self.enterz[key].delete(0)
            self.enterz[key].insert(0,expInfo[key])

            self.labelz[key][0].grid(row=self.labelz[key][1],column=self.labelz[key][2],sticky="E")
            self.enterz[key].grid(row=self.labelz[key][1],column=self.labelz[key][2]+1,sticky="W")
        
        # chooseDirOpts defines options for choosing the data directory to which one will save.
        self.chooseDirOpts = {}
        self.chooseDirOpts['initialdir'] = expInfo['AFS_Directory']
        self.chooseDirOpts['mustexist'] = False
        self.chooseDirOpts['parent'] = self
        self.chooseDirOpts['title'] = 'Choose the AFS base data directory...'
        
        # buttonz contains all of the buttons. Note the save button is disabled until the user
        #   chooses their save directory in self.askDirectory
        self.buttonz = {}
        self.buttonz['setupBNC'] = tk.Button(master=self,text='Setup NI BNC Box...',command=self.setupBNC)
        self.buttonz['chooseDir'] = tk.Button(master=self,text="Choose Data Directory...",command=self.askDirectory)
        self.buttonz['save'] = tk.Button(master=self,text="Save Data",state="disabled")
        self.buttonz['average'] = tk.Button(master=self,text="Start Averaging",command=self.startAverage)
        self.buttonz['resetAverage'] = tk.Button(master=self,text="Reset Average",command=self.resetAverage)
        self.buttonz['stopExperiment'] = tk.Button(master=self,text="Stop Experiment",command=self.stopCollection)
        
        self.buttonz['setupBNC'].grid(row=0,column=0,columnspan=4,sticky="NSEW")
        self.buttonz['chooseDir'].grid(row=11,column=0,columnspan=2)
        self.buttonz['save'].grid(row=11,column=2,rowspan=1,columnspan=2)
        self.buttonz['average'].grid(row=10,column=0,columnspan=2)
        self.buttonz['resetAverage'].grid(row=10,column=2,columnspan=2)
        self.buttonz['stopExperiment'].grid(row=12,column=2,columnspan=2)
        
        # Set resizing weights for the grid when the window changes size.
        # A weight of "0" implies that the row/column will not resize.
        # A weight of N implies that the row/column will resize at a rate of 
        #    factor N larger than the smallest weight (which is 1 here).
        for N in range(6):
            if N < 2:
                self.columnconfigure(N,weight=0)
            else:
                self.columnconfigure(N,weight=1)
            
        for N in range(12):
            if N < 8:
                self.rowconfigure(N,weight=0)
            elif N == 8:
                self.rowconfigure(N,weight=100)
            else:
                self.rowconfigure(N,weight=1)
                
        # Initialize the output of the wave
        self.initializeWaveOutput()
        
        # Commence updateWindow loop after giving it 100ms to chill out.
        self.after(100,self.updateWindow)
    
    def stopCollection(self):
        self.running = False
        
        
    def initializeWaveOutput(self):
        
        # Create the output task
        self.waveOutput = bnc.NIOutputWave(expInfo['bncVout'],expInfo['sampRate'])
        
        # Define the output wave
        self.waveOutput.setWave(genWave())
        
        # Start the output
        # print "started wave output"
        self.waveOutput.startWave()
    
    def initializeWaveInput(self):
        
        # Create the input task
        self.waveInput = bnc.NIReadWaves([expInfo['bncXin'],expInfo['bncYin']],expInfo['sampRate'])
        
    def getNewData(self):
        return self.waveInput.readWaves(int(expInfo['nSamps']))
    
    def setupBNC(self):
        
    
        
        def saveConf():
            for key in t.labelz:
                self.oldConf[key] = expInfo[key]
                expInfo[key] = t.enterz[key].get()
                if expInfo[key] is not self.oldConf[key]:
                    self.bncMarkedUpdate = True
            
            t.destroy()
        
        t = tk.Toplevel(self)
        t.wm_title('Set BNC-2110 Ports')
        
        t.labelz = {}
        t.enterz = {}
        
        t.labelz['bncXin'] = [tk.Label(t,text="X AI Channel:"),0,0]
        t.labelz['bncYin'] = [tk.Label(t,text="Y AI Channel:"),1,0]
        t.labelz['bncVout'] = [tk.Label(t,text="Vout AO Channel:"),2,0]
        t.labelz['sampRate'] = [tk.Label(t,text="Sampling Rate (Hz):"),3,0]
        t.labelz['nSamps'] = [tk.Label(t,text="Number of Samples:"),4,0]
        
        for key in t.labelz:
            t.enterz[key] = tk.Entry(t,width="13",foreground="#000000")
            t.enterz[key].delete(0)
            t.enterz[key].insert(0,expInfo[key])
            
            t.labelz[key][0].grid(row=t.labelz[key][1],column=t.labelz[key][2],sticky="E")
            t.enterz[key].grid(row=t.labelz[key][1],column=t.labelz[key][2]+1,sticky="W")
        
        t.buttonz = {}
        
        t.buttonz['saveConf'] = tk.Button(t,text='Save and Close',command=saveConf)
        t.buttonz['saveConf'].grid(row=5,column=0,columnspan=2)
        
    def updateBNC(self):
        self.diff_entries = set(self.oldConf.items()) ^ set(expInfo)
    
     
    def _quit(self):
        """ called when the window is closed."""
        self.waveInput.endRead()
        self.waveOutput.endWave()
        self.quit()     # stops mainloop
        self.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate



# Primary application loop.
if __name__ == "__main__":

    app = MeasureIV(None)
    app.title("Measure IV")
    app.mainloop()
