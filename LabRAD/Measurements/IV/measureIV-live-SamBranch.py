# -*- coding: utf-8 -*-
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
from math import pow, factorial
import os

# scipy packages
from pylab import *
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

# GUI packages
import Tkinter as tk
import ttk
import tkFileDialog
import tkMessageBox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#for tex maybe later
#from matplotlib.pyplot import plt

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
REFRESH_INTERVAL = 1 # in seconds can be float but flags unimportant error
PLOT_REFRESH_INTERVAL = 1 # in seconds


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
    # Currently is 0 and 1 so all wave output is 0's
    amp = float(expInfo['outputWaveAmp'])
    freq = float(expInfo['outputWaveFreq'])
    
    # Setting freq to 0 is just a DC output.
    # Number of samples doesn't really matter in
    #   that case, so just set to the sample rate.
    if freq is not 0:
        samps = int(float(expInfo['sampRate']) / freq)
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
        self.topLevelUp=False
        self.oldConf = {}
        self.topLevel = {'labelz' : {}, 'selects' : {},'enterz' : {}}
        
        #False if you don't want to plot the averages
        self.plottingAvgs = True
        
        #used to determine how many initial sample points there will be for clustering
        self.sampleSize = 180
        
        self.inputPars = ['bncXin','bncYin','sampRate']

        self.lock = threading.Lock()
        self.cond = threading.Condition(threading.Lock())

    def updateWindow(self):
        #have pop up data connected to picture
        self.oldWave = [expInfo['outputWaveAmp'],expInfo['outputWaveFreq']]

        for key in self.enterz:
            expInfo[key] = self.enterz[key].get()
        
        if self.saveDirChosen == True:
            setDataDir()
            self.directoryLabel.config(text="Save Path: " + expInfo['saveFile'],
                                       anchor="w",relief="groove",fg = "black")
        # Check if the BNC settings have changed since last update loop
        #     If they have, kill old handles and open new ones.
        #       needed to change the below conditionals
        if self.bncMarkedUpdate:
            self.bncMarkedUpdate = False
            
            if not (self.oldConf['bncVout'] == expInfo['bncVout'] and self.oldConf['sampRate'] == expInfo['sampRate']):
                self.waveOutput.endWave()
                self.oldConf['bncVout'] = expInfo['bncVout']
                self.oldConf['sampRate'] = expInfo['sampRate']
                self.initializeWaveOutput()
            
            if not (self.oldConf['bncXin'] == expInfo['bncXin'] and self.oldConf['bncYin'] == expInfo['bncYin'] and self.oldConf['sampRate'] == expInfo['sampRate']):
                self.waveInput.endRead()
                self.oldConf['bncXin'] = expInfo['bncXin']
                self.oldConf['bncYin'] = expInfo['bncYin']
                self.oldConf['sampRate'] = expInfo['sampRate']
                self.initializeWaveInput()
            
        # Check if a new wave needs to be written to the BNC device.
        if not (self.oldWave[0] == expInfo['outputWaveAmp'] and self.oldWave[1] == expInfo['outputWaveFreq']):
            #don't go higher than 20 or 1ish Hz
            
            if(int(expInfo['outputWaveFreq']) > 10):
            # If it does, reinitialize it.
                self.waveOutput.endWave()
                self.initializeWaveOutput()
        
        # Initialize plot loop and re-run the updateWindow method every REFRESH_INTERVAL ms.
        if self.updatingPlot == False:
            self.updatePlot()
        
        self.after(REFRESH_INTERVAL,self.updateWindow)
    
    def updatePlot(self):
        self.updatingPlot = True
        self.avgStart = 0
        
        self.data_read = threading.Thread(target=self.collect_data, args=())
        
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
            
            self.pltcanvas = FigureCanvasTkAgg(self.fig,master=self.leftFrame)
            self.pltcanvas.show()
            self.pltcanvas.get_tk_widget().grid(row=8,column=0,rowspan=1,columnspan=4)
            
            self.line = self.ax.plot([],[],'r-')[0]           
            self.initializeWaveInput()
            
            print "starting threads"
            self.data_read.start()
        
        return
        
    def collect_data(self):

        while (self.running):
            sleep(REFRESH_INTERVAL)
            self.cond.acquire()
            print "getting data"
            
            newdata = self.getNewData()
            #this is dummy data. Uncomment the line above
            #newdata = [10 * np.random.random_sample(10),10* np.random.random_sample(10)]
            
            list = newdata
            list[0] = np.array(list[0])
            list[1] = np.array(list[1])
            
            #specific calculations here
            #I/V = X/Y I think
            if(self.tab_mode == 1):
                r_i = self.tabs[1]['enterbuttons']['r-i']['entry'].get()
                r_v = self.tabs[1]['enterbuttons']['r-v']['entry'].get()
                amp = self.tabs[1]['enterbuttons']['gain-a']['entry'].get()
                
                #replace 1 with expected default
                r_i = float(r_i) if self.is_number(r_i) else 1 #default
                r_v = float(r_v) if self.is_number(r_v) else 1 #default
                amp = float(amp) if self.is_number(amp) else 1 #default
                
                list[0] = list[0]
                list[1] = list[1] * amp

            elif(self.tab_mode == 3):     
                r_i = self.tabs[3]['enterbuttons']['r-i']['entry'].get()
                r_v = self.tabs[3]['enterbuttons']['r-v']['entry'].get()
                amp = self.tabs[3]['enterbuttons']['gain-a']['entry'].get()
                r_coil = self.tabs[3]['enterbuttons']['r-coil']['entry'].get()
                
                #replace 1 with expected default
                r_i = float(r_i) if self.is_number(r_i) else 1 #default
                r_v = float(r_v) if self.is_number(r_v) else 1 #default
                amp = float(amp) if self.is_number(amp) else 1 #default
                r_coil = float(r_coil) if self.is_number(r_coil) else 1 #default
                
                list[0] = list[0]
                list[1] = list[1] * amp

            list[0].tolist()
            list[1].tolist()
                
            if self.flip.get() == 0:
                self.line.set_xdata(list[0])
                self.line.set_ydata(list[1])     
            else:
                self.line.set_xdata(list[1])
                self.line.set_ydata(list[0])
            
            self.read_ready = True
            self.cond.notify()
            self.cond.release()
            
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            
        self._quit()
        return
        
    def startAverage(self):
        self.avgStart = 0

        e = self.enterz['nAvgs']
        e.config(state='disabled')
        self.nAvgs = int(e.get())
        
        avg = threading.Thread(target=self.average, args=())
        avg.start()
        #avg.join() #for some reason this screws everything up
        #return
    
    #fill 2d array with 0's first
    def binomial_coeff(self,n,k):
        i = self.stored_coeff[n][k]
        if i == 0:
            i = factorial(n) / (factorial(k) * factorial(n-k))
            self.stored_coeff[n][k] = i
        
        return i
    
    #this function is unused now but may come in handy later
    #if this method of interpolation is desired
    def bezier_curve(self, points):
        
        n = len(points) - 1
        p = []
        for param in range(500):
            t = param / 500.0
            sumx = 0
            sumy = 0
            for i in range(n + 1):
                m = pow((1 - t),(n - i)) * pow(t,i)
                sumx = sumx + self.binomial_coeff(n,i) * m * points[i][0]
                sumy = sumy + self.binomial_coeff(n,i) * m * points[i][0]
            p.append([sumx, sumy])
            
        return p
 
    def orderPoints(self, points, base_min):
        pts = []
        order = []
        total_dist = 0
        for p in points:
            pts.append({'point' : p, 'next': None, 'previous' : False, 'dist' : base_min})
            
        l = range(len(pts))
        index = 0
        pts[index]['previous'] = True
        for i in l:
            dist = base_min
            p = pts[index]
            next_ind = -1
            
            for j in l:
                if index == j:
                    continue
                
                q = pts[j]
                if not q['previous']: 
                    d = (q['point'][0] - p['point'][0])**2 + (q['point'][1] - p['point'][1])**2
                    if d <= dist:
                        dist = d
                        next_ind = j

            if next_ind == -1:
                print 'No suitable point found'
                break
                
            order.append(next_ind)
            total_dist += dist
            pts[index]['next'] = next_ind
            pts[index]['dist'] = dist
            pts[next_ind]['previous'] = True
            index = next_ind
            
            if i == 2:
                pts[0]['previous'] = False
    
        if len(order) == 0:
            return []
            
        good_dist = 2 * (total_dist / len(order))
        points = []
        index = 0
        for o in order:
            #can't just not add points to avoid this problem
            #happends when points are too far away at the end of the process
            #if pts[o]['dist'] <= good_dist:
            points.append(pts[o]['point'])

        return points
    
    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def average(self):
        self.graphs = range(self.nAvgs)
        self.averaging = True
        e = self.enterz['currentAverages']
        max_a = 0   #for display
        
        while(self.running):         
            self.cond.acquire()
            
            #wait for data to be read in
            while(not self.read_ready):
                self.cond.wait()

            self.read_ready = False            
            if(not self.averaging):
                self.cond.release()
                break;
            
            self.graphs[self.avgStart] = self.line.get_xydata()           
            self.avgStart = self.avgStart + 1   
            #we could do a queue to get all the data points but we don't care too much  
            
            if(self.avgStart > max_a):
                max_a = self.avgStart        
            if(self.avgStart == self.nAvgs):
                self.avgStart = 0
            
            END = len(e.get())
            e.config(state='normal')
            e.delete(0, END)
            e.insert(0,str(max_a))
            e.config(state='disabled')
            self.cond.release()
            
        e = self.enterz['nAvgs']
        e.config(state='normal')

        #now we can actually average
        self.calcAverage()
        return
    
    def calcAverage(self):
        x_axis = []
        y_axis = []
        g_len = len(self.graphs)
        #x and y are switched in the get_xydata() method
        
        if g_len < 1:
            return
        #calculate lowest possible amount of points
        min = 10000

        for i in range(g_len):
            g = self.graphs[i]
            min = len(g) if len(g) < min else min
            
        print '\nmin: ' + str(min) + ' points per sample\n'
        
        Xmax = -10000
        Xmin = 10000
        Ymax = -10000
        Ymin = 10000
        points = []
        tmpPoints = []
        d_sum = 0
        
        #remove points too far away from the next first
        for i in range(min):
            sumx = 0
            sumy = 0
            for j in range(g_len):  
                '''
                if self.graphs[j][i][0] > Xmax:
                    Xmax = self.graphs[j][i][0]
                if self.graphs[j][i][0] < Xmin:
                    Xmin = self.graphs[j][i][0]
                if self.graphs[j][i][1] > Ymax:
                    Ymax = self.graphs[j][i][1]
                if self.graphs[j][i][1] < Ymin:
                    Ymin = self.graphs[j][i][1]  
                tmpPoints.append([self.graphs[j][i][0],self.graphs[j][i][1]])
                
                if i < min - 1:
                    d_sum += (self.graphs[j][i][0] - self.graphs[j][i + 1][0])**2 + (self.graphs[j][i][1] - self.graphs[j][i + 1][1])**2 
                '''
                sumx += self.graphs[j][i][0]
                sumy += self.graphs[j][i][1]
            x_axis.append(sumx / g_len)
            y_axis.append(sumy / g_len)
        '''
        d_avg = (d_sum * 1.0) / len(tmpPoints)   
        d_avg *= 1.3
         
        for i in range(len(tmpPoints) - 1):
            d = ((tmpPoints[i][0] - tmpPoints[i + 1][0])**2 + (tmpPoints[i][1] - tmpPoints[i + 1][1])**2)
            if d_avg <= d:
                #print d
                points.append(tmpPoints[i])
    
        print str(len(points)) + "total points"
        #might want to change 500
        l = range(500)
        sel = int(len(points) / 500)
        means = []
        
        for i in l:
            ind = int(i * sel * np.random.rand())
            means.append(points[ind])
        base_min = max((Xmax - Xmin) / 1.0,(Ymax - Ymin) / 1.0)**2
        print base_min
                 
        p_min = 0
        dist = 0 
        
        #clustering 1000+ many points takes way too long
        #tries to converge after 2 iterations
        for iter in range(2):#(10):
            mean_dict = {}
            for p in points:
                p_min = base_min
                m_index = -1
                i = 0
                for m in means:
                    dist = (p[0] - m[0])**2 + (p[1] - m[1])**2
                   
                    if dist <= p_min:
                        p_min = dist
                        m_index = i     
                    i += 1
                    
                if mean_dict.has_key(m_index):
                    mean_dict[m_index].append(p)
                else:
                    mean_dict[m_index] = [p]
            
            print "refitting"
            
            s = 0
            means = []
            for i in mean_dict:
                x_sum = 0
                y_sum = 0
                for j in mean_dict[i]:
                    x_sum += j[0]
                    y_sum += j[1]
                
                if(len(mean_dict[i]) > 0):
                    s += 1
                    means.append([x_sum / len(mean_dict[i]), y_sum / len(mean_dict[i])])
            print s
        
        print "ordering points..."
        #result now stores the average line
        #need to reorder the points first since we only have an unordered set of cluster points       
        means = self.orderPoints(means, base_min) #discard edges twice as large as average
        self.stored_coeff = np.zeros([len(means), len(means)])
        #might want to comment out below
        means = self.bezier_curve(means)
        means = np.rot90(means)
        self.result = [means[1], means[0]]
        '''
        
        self.result = [x_axis, y_axis]
        self.avgLine = self.ax.plot([],[],'b-')[0] 
        
        #this draws the average in blue and the ones it took in green
        if(self.plottingAvgs):
            self.g_line = {}
            '''
            for i in range(g_len):
                self.g_line[i] = self.ax.plot([],[],'g-')[0]
                g_x = []
                g_y = []
                for j in range(min):
                    g_x.append(self.graphs[i][j][0])
                    g_y.append(self.graphs[i][j][1])
                self.g_line[i].set_xdata(g_x)
                self.g_line[i].set_ydata(g_y)
            '''  
            self.avgLine = self.ax.plot([],[],'b-')[0] 
            self.avgLine.set_xdata(self.result[0])
            self.avgLine.set_ydata(self.result[1])         
        
        return
    
    def resetAverage(self):
        # A running average is now reset and starts over.
        self.averaging = False
       
        self.avgStart = 0
        self.graphs = []    
        
        if(self.plottingAvgs):
            self.ax.lines.remove(self.avgLine)
            for i in self.g_line:
                self.ax.lines.remove(self.g_line[i])
                
        print 'reset'
        return
    
    def stopAverage(self):
        self.averaging = False
        
    def saveData(self):
        
        for key in self.enterz:
            expInfo[key] = self.enterz[key].get()

        data = "COMMENTS: "+str(expInfo['comments'])+"\n"
        if self.flip.get() == 0:
            data += "Voltage\t\t\t Current\n"
        else:
            data += "Current\t\t\t Voltage\n"
        data += "-------------------------------------\n"
        for i in range(len(self.result[0])):
            data += str(self.result[0][i]) + "\t" + str(self.result[1][i]) + "\n"
        
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
        #give x/y channels a dropdown 0-7 in y has output option which sets output of (below)
        #and output Vout AO Channel (out) 0-7
        
        # This method writes the GUI for display. It is only called once at program startup.
        
        self.leftFrame = tk.Frame(self)
        self.leftFrame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.rightFrame = tk.Frame(self, width=600)
        self.rightFrame.pack(side=tk.LEFT)
        
        # self.headerLabel = tk.Label(self, text="Measure IV")
        self.directoryLabel = tk.Label(self.leftFrame, text="PLEASE CHOOSE DATA DIRECTORY",relief="groove",fg="red")
        
        # self.headerLabel.grid(row = 0, column = 0, columnspan = 6)
        self.directoryLabel.grid(row=12,column=0,columnspan=4)
        
        
        
        # Define a dictionary filled with [tk.Labels, row_num, col_num] to generate
        #    labels and entry fields usable by the user.
        
        self.enterz = {}
        self.labelCtrl = {}
        self.labelCtrl['measurementType'] = [tk.Label(self.leftFrame,text="Measurement Type:"),1,0]
        self.labelCtrl['deviceType'] = [tk.Label(self.leftFrame,text="Device Type:"),2,0]
        self.labelCtrl['waferName'] = [tk.Label(self.leftFrame,text="Wafer Name:"),2,1]
        self.labelCtrl['dieName'] = [tk.Label(self.leftFrame,text="Die Name:"),2,2]
        self.labelCtrl['measurementLocation'] = [tk.Label(self.leftFrame,text="Measurement Loc:"),2,3]
        self.labelCtrl['outputWaveFreq'] = [tk.Label(self.leftFrame,text="Output Wave Frequency (Hz):"),5,2]
        
        for key in self.labelCtrl:
            self.enterz[key] = tk.Entry(self.leftFrame,width="10")
            self.enterz[key].delete(0)
            self.enterz[key].insert(0,expInfo[key])
     
            self.labelCtrl[key][0].grid(row=self.labelCtrl[key][1],column=self.labelCtrl[key][2],sticky=tk.W)
            self.enterz[key].grid(row=self.labelCtrl[key][1],column=self.labelCtrl[key][2],sticky=tk.E)
    
        #self.flip_axes_l = tk.Label(self.leftFrame,text="Flip axes")
        #self.flip_axes_l.grid(row=1,column=2,sticky=tk.W)
        self.flip = tk.IntVar();
        self.flip_axes = tk.Checkbutton(self.leftFrame,text="Flip axes", variable=self.flip)
        self.flip_axes.grid(row=1,column=2,sticky=tk.W)
        self.flip.set(0)
            
        self.grid_columnconfigure(0,minsize="170")
        self.grid_columnconfigure(1,minsize="140")
        self.grid_columnconfigure(2,minsize="150")
        self.grid_columnconfigure(3,minsize="150")

        self.enterz['comments'] = tk.Entry(self.leftFrame,width="110")
        self.enterz['comments'].delete(0)
        self.enterz['comments'].insert(0,expInfo['comments'])
        self.enterz['comments'].grid(row=3,column=1,rowspan=2,columnspan=3)
        tk.Label(self.leftFrame,text="Comments:").grid(row=3,column=0)
            
        self.labelz = {}
        self.labelz['outputWaveAmp'] = [tk.Label(self.leftFrame,text="Output Wave Amplitude (V):"),5,0]        
        self.labelz['nAvgs'] = [tk.Label(self.leftFrame,text="Averages:"),9,0]
        self.labelz['currentAverages'] = [tk.Label(self.leftFrame,text="Current Averages:"),9,2]

        # Loop over all labelz and create each with an associated entry field.
        for key in self.labelz:
            self.enterz[key] = tk.Entry(self.leftFrame,width="13")
            self.enterz[key].delete(0)
            self.enterz[key].insert(0,expInfo[key])
            
            self.labelz[key][0].grid(row=self.labelz[key][1],column=self.labelz[key][2],sticky="E")
            self.enterz[key].grid(row=self.labelz[key][1],column=self.labelz[key][2]+1,sticky="W")
       
        e = self.enterz['currentAverages']
        e.config(state='disabled')
        
        # chooseDirOpts defines options for choosing the data directory to which one will save.
        self.chooseDirOpts = {}
        self.chooseDirOpts['initialdir'] = expInfo['AFS_Directory']
        self.chooseDirOpts['mustexist'] = False
        self.chooseDirOpts['parent'] = self
        self.chooseDirOpts['title'] = 'Choose the AFS base data directory...'
        
        # buttonz contains all of the buttons. Note the save button is disabled until the user
        #   chooses their save directory in self.askDirectory
        
        #add stop averaging button
        self.buttonz = {}
        self.buttonz['setupBNC'] = tk.Button(master=self.leftFrame,text='Setup NI BNC Box...',command=self.setupBNC)
        self.buttonz['chooseDir'] = tk.Button(master=self.leftFrame,text="Choose Data Directory...",command=self.askDirectory)
        self.buttonz['save'] = tk.Button(master=self.leftFrame,text="Save Data",state="disabled")
        self.buttonz['average'] = tk.Button(master=self.leftFrame,text="Start Averaging",command=self.startAverage)
        self.buttonz['resetAverage'] = tk.Button(master=self.leftFrame,text="Reset Average",command=self.resetAverage)
        self.buttonz['stopAverage'] = tk.Button(master=self.leftFrame,text="Stop Average",command=self.stopAverage)
        
        self.buttonz['setupBNC'].grid(row=0,column=0,columnspan=4,sticky="NSEW")  
        self.buttonz['average'].grid(row=10,column=0, columnspan=2)
        self.buttonz['resetAverage'].grid(row=10,column=2)
        self.buttonz['stopAverage'].grid(row=10,column=3)
        self.buttonz['chooseDir'].grid(row=11,column=0, columnspan=2)
        self.buttonz['save'].grid(row=11,column=2)
        #self.buttonz['stopExperiment'].grid(row=11,column=3)
        
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
                
        #tabs:
        self.grid_columnconfigure(5,minsize="140")
        self.grid_columnconfigure(6,minsize="140")
        self.grid_columnconfigure(7,minsize="140")
        self.grid_columnconfigure(8,minsize="140")
        
        '''
        self.tabs = [{},{},{},{}]
        self.tabs[0]['button'] = tk.Button(master=self,text='4-Wire IV',command=lambda: self.switchTab(0))
        self.tabs[0]['button'].grid(row=0,column=5,sticky="NSEW")
        self.tabs[0]['image'] = Image.open("C:\Users\Public\Pictures\Sample Pictures\Desert.jpg")
        self.tabs[0]['photo'] = ImageTk.PhotoImage(self.tabs[0]['image'])
        self.tabs[0]['enterbuttons'] = {
        #    'r-in' : {'entry' : tk.Entry(self,width="13"), 'x' : 120, 'y' : 30},
        #    'r-out' : {'entry' : tk.Entry(self,width="13"), 'x' : 120, 'y' : 30},
        #    'rv-in' : {'entry' : tk.Entry(self,width="13"), 'x' : 120, 'y' : 30},
        #    'rv-out' : {'entry' : tk.Entry(self,width="13"), 'x' : 120, 'y' : 30}
        }
        self.tabs[0]['label'] = tk.Label(image=self.tabs[0]['photo'])
        self.tabs[0]['label'].grid(row=1,column=5,sticky="NS",rowspan=10,columnspan=4)
        
        self.tabs[1]['button'] = tk.Button(master=self,text='3-Wire IV',command=lambda: self.switchTab(1))
        self.tabs[1]['button'].grid(row=0,column=6,sticky="NSEW")
        self.tabs[1]['image'] = Image.open("Z:\mcdermott-group\QBITElectronics\LabRAD-SLUG\scripts\SLUG\ThreeWire.gif")
        self.tabs[1]['photo'] = ImageTk.PhotoImage(self.tabs[1]['image'])
        self.tabs[1]['enterbuttons'] = {
            'r-i' : {'entry' :  None, 'type': 'ent', 'x' : 965, 'y' : 235},
            'r-v' : {'entry' : None, 'type': 'ent', 'x' : 1265, 'y' : 235},
            'gain-a' : {'entry' : None, 'type': 'ent', 'x' : 1270, 'y' : 100}
        }
        self.tabs[1]['label'] = tk.Label(image=self.tabs[1]['photo'])
        self.tabs[1]['label'].grid(row=1,column=5,sticky="NS",rowspan=10,columnspan=4)
        self.tabs[1]['label'].grid_remove()
        
        
        self.tabs[2]['button'] = tk.Button(master=self,text='2-Wire IV',command=lambda: self.switchTab(2))
        self.tabs[2]['button'].grid(row=0,column=7,sticky="NSEW")
        self.tabs[2]['image'] = Image.open("C:\Users\Public\Pictures\Sample Pictures\Penguins.jpg")
        self.tabs[2]['photo'] = ImageTk.PhotoImage(self.tabs[2]['image'])
        self.tabs[2]['enterbuttons'] = {
        #    'r' : {'entry' : tk.Entry(self,width="13"), 'x' : 120, 'y' : 30}
        }
        self.tabs[2]['label'] = tk.Label(image=self.tabs[2]['photo'])
        self.tabs[2]['label'].grid(row=1,column=5,sticky="NS",rowspan=10,columnspan=4)
        self.tabs[2]['label'].grid_remove()
        
        
        self.tabs[3]['button'] = tk.Button(master=self,text="V - Φ",command=lambda: self.switchTab(3))
        self.tabs[3]['button'].grid(row=0,column=8,sticky="NSEW")
        self.tabs[3]['y-axis'] = "Φ/M[Φ_o/H]"
        self.tabs[3]['image'] = Image.open("Z:\mcdermott-group\QBITElectronics\LabRAD-SLUG\scripts\SLUG\VPhi.gif")
        self.tabs[3]['photo'] = ImageTk.PhotoImage(self.tabs[3]['image'])
        self.tabs[3]['enterbuttons'] = {
            'r-i' : {'entry' : None, 'type': 'ent', 'x' : 965, 'y' : 235},
            'r-v' : {'entry' : None, 'type': 'ent', 'x' : 1265, 'y' : 235},
            'gain-a' : {'entry' : None, 'type': 'ent', 'x' : 1270, 'y' : 100},
            'r-coil' : {'entry' : None, 'type': 'ent', 'x' : 905, 'y' : 410},
            
            'bncXin' : {'entry' : None, 'type': 'sel', 'opt' : range(8), 'x' : 895, 'y' : 150},  #i-term
            'bncVout' : {'entry' : None, 'type': 'sel', 'opt' : range(8), 'x' : 1420, 'y' : 150} #v-term
        }
        self.tabs[3]['label'] = tk.Label(image=self.tabs[3]['photo'])
        self.tabs[3]['label'].grid(row=1,column=5,sticky="NS",rowspan=10,columnspan=4)
        self.tabs[3]['label'].grid_remove()
        
        for i in range(4):
            for j in self.tabs[i]['enterbuttons']:
                c = self.tabs[i]['enterbuttons'][j]
                if c['type'] == 'ent':
                    self.tabs[i]['enterbuttons'][j]['sv'] = tk.StringVar(self)
                    self.tabs[i]['enterbuttons'][j]['sv'].trace("w", lambda name, index, mode, var=self.tabs[i]['enterbuttons'][j], j=j : self.changeAxes(j,i))
                    c['entry'] = tk.Entry(self,width="10",bd = 3,textvariable=c['sv'])
                elif c['type'] == 'sel':
                    self.tabs[i]['enterbuttons'][j]['sv'] = tk.StringVar(self)
                    self.tabs[i]['enterbuttons'][j]['sv'].set(c['opt'][0])
                    self.tabs[i]['enterbuttons'][j]['sv'].trace("w", lambda name, index, mode, i=i, j=j: self.changeSelection(i,j))
                    c['entry'] = apply(tk.OptionMenu, (self, self.tabs[i]['enterbuttons'][j]['sv']) + tuple(c['opt']))
                    c['entry'].config(width=9,bg='white')
        
        self.tab_mode = 0
        self.switchTab(1)
        '''
        
        ### RIGHT SIDE ###
        self.measurementTabs = ttk.Notebook(self.rightFrame, width=600)
        self.measurementTabs.pack(side=tk.TOP)
        frame2wire = ttk.Frame(self.measurementTabs)
        frame3wire = ttk.Frame(self.measurementTabs)
        frame4wire = ttk.Frame(self.measurementTabs)
        frameVPhi  = ttk.Frame(self.measurementTabs)
        self.measurementTabs.add(frame2wire, text='2-Wire')
        self.measurementTabs.add(frame3wire, text='3-Wire')
        self.measurementTabs.add(frame4wire, text='4-Wire')
        self.measurementTabs.add(frameVPhi, text='V-Phi')
        
        #bgimg = tk.PhotoImage(file="TwoWire.gif")
        #bglabel2 = tk.Label(frame2wire, image=bgimg)
        #bglabel2.image = bgimg
        #bglabel2.pack()
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
        
        self.RACIn = tk.DoubleVar()
        self.RDCIn = tk.DoubleVar()
        self.ROut = tk.DoubleVar()
        self.portACIn = tk.IntVar()
        self.portDCIn = tk.IntVar()
        self.portOut = tk.IntVar()
        self.amp = tk.DoubleVar()
        self.ACFreq = tk.DoubleVar()
        self.ACAmp = tk.DoubleVar()
        self.sampRate = tk.IntVar()
        self.nSamples = tk.IntVar()
        self.RACIn.set(100)
        self.RDCIn.set(100)
        self.ROut.set(100)
        self.portACIn.set(0)
        self.portDCIn.set(1)
        self.portOut.set(0)
        self.amp.set(1)
        self.ACFreq.set(1)
        self.ACAmp.set(0)
        self.sampRate.set(10000)
        self.nSamples.set(10000)
        
        tk.OptionMenu(frame3wire, self.portACIn, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=93/597., rely=58/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ACFreq).place(relx=93/597., rely=81/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ACAmp).place(relx=93/597., rely=105/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.RACIn).place(relx=144/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.ROut).place(relx=405/597., rely=176/578., anchor=tk.CENTER)
        tk.Entry(frame3wire, width=8, textvariable=self.amp).place(relx=411/597., rely=35/578., anchor=tk.CENTER)
        tk.OptionMenu(frame3wire, self.portOut, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=545/597., rely=80/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frame4wire, self.portACIn, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=58/597., rely=158/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ACFreq).place(relx=58/597., rely=182/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ACAmp).place(relx=58/597., rely=205/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.RACIn).place(relx=38/597., rely=268/578., anchor=tk.CENTER)
        tk.Entry(frame4wire, width=8, textvariable=self.ROut).place(relx=220/597., rely=268/578., anchor=tk.CENTER)
        tk.OptionMenu(frame4wire, self.portOut, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=551/597., rely=94/578., anchor=tk.CENTER)
        
        tk.OptionMenu(frameVPhi, self.portDCIn, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=94/597., rely=80/578., anchor=tk.CENTER)
        tk.OptionMenu(frameVPhi, self.portACIn, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=44/597., rely=192/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ACFreq).place(relx=44/597., rely=216/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ACAmp).place(relx=44/597., rely=239/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.RDCIn).place(relx=144/597., rely=156/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.RACIn).place(relx=94/597., rely=306/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.ROut).place(relx=405/597., rely=156/578., anchor=tk.CENTER)
        tk.Entry(frameVPhi, width=8, textvariable=self.amp).place(relx=411/597., rely=36/578., anchor=tk.CENTER)
        tk.OptionMenu(frameVPhi, self.portOut, 0,1,2,3,4,5,6,7, command=lambda i=0: self.saveConf()).place(relx=545/597., rely=80/578., anchor=tk.CENTER)
        
        self.tab_mode = 0
        self.measurementTabs.select(1)
        self.protocol("WM_DELETE_WINDOW", self._quit) #X BUTTON
        
        # Initialize the output of the wave
        self.initializeWaveOutput()
        
        # Commence updateWindow loop after giving it 100ms to chill out.
        self.after(100,self.updateWindow)
    
    def stopCollection(self):
        self.running = False
    
    def switchTab(self,tab):
        for i in range(4):
            self.tabs[i]['label'].grid_remove()
            
        for j in self.tabs[self.tab_mode]['enterbuttons']:
            c = self.tabs[self.tab_mode]['enterbuttons'][j]
            c['entry'].grid(row=6,column=6)
            c['entry'].grid_remove()
            
        for j in self.tabs[tab]['enterbuttons']:
            c = self.tabs[tab]['enterbuttons'][j]
            c['entry'].grid(row=6,column=6)
            c['entry'].place(x = c['x'], y = c['y'])
            c['entry'].grid_remove()
                
        self.tabs[tab]['label'].grid()
        self.tab_mode = tab

        
    #callback for entry boxes in case we need it
    #I made this and realized I didn't need it but oh well
    def changeAxes(self, entry, tab):
        print str(entry) + ' changed'
        
    #obsolete
    def changeSelection(self, tab, index):
        # be sure index in the enterbuttons array is the 
        # same index as that on the expInfo 
        val = int(self.tabs[tab]['enterbuttons'][index]['sv'].get())
        self.saveConf()
#        print self.topLevel
#        self.oldConf[index] = expInfo[index]
#        expInfo[index] = val
#        self.bncMarkedUpdate = True
        
    def initializeWaveOutput(self):
        try:
            self.waveOutput = bnc.NIOutputWave(expInfo['bncVout'],expInfo['sampRate'])
            self.waveOutput.setWave(genWave())
            print "started wave output"
            self.waveOutput.startWave()
        except Exception as e:
            e = sys.exc_info()[0]
            print 'Error initializing wave output:\n' + str(e)
    
    def initializeWaveInput(self):
        
        # Create the input task
        self.waveInput = bnc.NIReadWaves([expInfo['bncXin'],expInfo['bncYin']],expInfo['sampRate'])
        
    def getNewData(self):
        #data interval = output freq
        return self.waveInput.readWaves(int(expInfo['nSamps']))
    
    def close_tl(self):
        self.topLevelUp = False
    
    def setupBNC(self):
        
        self._tl = tk.Toplevel(self)
        self._tl.wm_title('Set BNC-2110 Ports')
        self._tl.protocol("WM_DELETE_WINDOW",self.close_tl())
        self.topLevelUp=True     
        
        #should maybe separate these declarations and put them into initialize()
        #then make them appear when this function is called. This way they retain their values
        self.topLevel['selects']['bncXin'] = {'desc' : "X AI Channel:",'x' : 0,'y' : 0,'opt' : range(8)}
        self.topLevel['selects']['bncYin'] = {'desc' : "Y AI Channel:",'x' : 1,'y' : 0,'opt' : range(8)}
        self.topLevel['selects']['bncVout'] = {'desc' : "Vout AO Channel:",'x' : 2,'y' : 0,'opt' : range(8)}
        
        self.topLevel['labelz']['sampRate'] = [tk.Label(self._tl,text="Sampling Rate (Hz):"),3,0]
        self.topLevel['labelz']['nSamps'] = [tk.Label(self._tl,text="Number of Samples:"),4,0]
        
        for key in self.topLevel['labelz']:
            self.topLevel['enterz'][key] = tk.Entry(self._tl,width="13",foreground="#000000")
            self.topLevel['enterz'][key].delete(0)
            self.topLevel['enterz'][key].insert(0,expInfo[key])
            
            self.topLevel['labelz'][key][0].grid(row=self.topLevel['labelz'][key][1],column=self.topLevel['labelz'][key][2],sticky="E")
            self.topLevel['enterz'][key].grid(row=self.topLevel['labelz'][key][1],column=self.topLevel['labelz'][key][2]+1,sticky="W")
        
        for i in self.topLevel['selects']:
            self.topLevel['selects'][i]['sv'] = tk.StringVar(self)
            self.topLevel['selects'][i]['sel'] = apply(tk.OptionMenu, (self._tl, self.topLevel['selects'][i]['sv']) + tuple(self.topLevel['selects'][i]['opt']))
            self.topLevel['selects'][i]['sel'].config(width=9,bg='white')
            tk.Label(self._tl,text=self.topLevel['selects'][i]['desc']).grid(row=self.topLevel['selects'][i]['x'],column=self.topLevel['selects'][i]['y'],sticky="W")
            self.topLevel['selects'][i]['sel'].grid(row=self.topLevel['selects'][i]['x'],column=self.topLevel['selects'][i]['y'] + 1,sticky="W")
       
        self.topLevel['selects']['bncXin']['sv'].set(expInfo['bncXin'])
        self.topLevel['selects']['bncYin']['sv'].set(expInfo['bncYin'])
        self.topLevel['selects']['bncVout']['sv'].set(expInfo['bncVout'])
       
        self.topLevel['buttonz'] = {}
        self.topLevel['buttonz']['saveConf'] = tk.Button(master=self._tl,text='Save and Close',command=lambda: self.saveConf())
        self.topLevel['buttonz']['saveConf'].grid(row=5,column=0,columnspan=2,sticky="NSWE")
    
    
    
    def saveConf(self):
        print 'saving configuration...'

        if self.topLevelUp:
            for key in self.topLevel['labelz']:
                self.oldConf[key] = expInfo[key]
                expInfo[key] = self.topLevel['enterz'][key].get()
                if expInfo[key] is not self.oldConf[key]:
                    self.bncMarkedUpdate = True
            
            for key in self.topLevel['selects']:
                self.oldConf[key] = expInfo[key]
                expInfo[key] = int(self.topLevel['selects'][key]['sv'].get())
                if expInfo[key] is not self.oldConf[key]:
                    if key == 'bncXin':
                        self.portACIn.set(expInfo[key])
                    if key == 'bncYin':
                        self.portDCIn.set(expInfo[key])
                    if key == 'bncVout':
                        self.portOut.set(expInfo[key])
                        
                    self.bncMarkedUpdate = True
                    
            self._tl.destroy()
            self.topLevelUp=False
        else:
            self.oldConf['sampRate'] = expInfo['sampRate']
            self.oldConf['nSamps'] = expInfo['nSamps']
            self.oldConf['bncXin'] = expInfo['bncXin']
            expInfo['bncXin'] = self.portACIn.get()
            self.oldConf['bncYin'] = expInfo['bncYin']
            expInfo['bncYin'] = self.portDCIn.get()
            self.oldConf['bncVout'] = expInfo['bncVout']
            expInfo['bncVout'] = self.portOut.get()
            self.bncMarkedUpdate = True
    
    def updateBNC(self):
        self.diff_entries = set(self.oldConf.items()) ^ set(expInfo)
    
     
    def _quit(self):
        """ called when the window is closed."""
        print 'quitting'
        self.waveInput.endRead()
        self.waveOutput.endWave()
        self.quit()     # stops mainloop
        self.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        os._exit(1)



# Primary application loop.
if __name__ == "__main__":
    app = MeasureIV(None)
    app.title("Measure IV")
    app.mainloop()