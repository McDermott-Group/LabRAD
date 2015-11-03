# Author: Ed Leonard
# Last updated: 13 February 2015


# Purpose: general functions for voltage input/output with NI BNC-2110

# Import useful libraries

""" C SYNTAX OF SELECTED DAQmx FUNCTIONS (in no particular order)

int32 DAQmxCreateTask (const char taskName[], TaskHandle *taskHandle);

int32 DAQmxCreateAOVoltageChan (TaskHandle taskHandle, const char physicalChannel[],
            const char nameToAssignToChannel[], float64 minVal, float64 maxVal, 
            int32 units, const char customScaleName[]);

int32 DAQmxWriteAnalogF64 (TaskHandle taskHandle, int32 numSampsPerChan, bool32 autoStart, 
            float64 timeout, bool32 dataLayout, float64 writeArray[], 
            int32 *sampsPerChanWritten, bool32 *reserved);

int32 DAQmxCfgSampClkTiming (TaskHandle taskHandle, const char source[], float64 rate, 
            int32 activeEdge, int32 sampleMode, uInt64 sampsPerChanToAcquire);

int32 DAQmxCreateAIVoltageChan (TaskHandle taskHandle, const char physicalChannel[], 
            const char nameToAssignToChannel[], int32 terminalConfig, float64 minVal, 
            float64 maxVal, int32 units, const char customScaleName[]);
            
int32 DAQmxReadAnalogF64 (TaskHandle taskHandle, int32 numSampsPerChan, float64 timeout, 
            bool32 fillMode, float64 readArray[], uInt32 arraySizeInSamps, 
            int32 *sampsPerChanRead, bool32 *reserved);
"""

# The PyDAQmx library is available here: http://pythonhosted.org/PyDAQmx/
try:
    from PyDAQmx import *
except ImportError:
    print "This program requires PyDAQmx to run."
    
import numpy as np

NI_BNC_MAX_VOLTAGE = 10.0;  # Max voltage output
NI_BNC_MAX_SAMP_RATE = 10000;   # Measurement/generation rate

# Function to change continuous voltage of single channel.
# Inputs: channel number (0:1) and voltage (-10:10)
def setOutputChanVolt(chan,outputVoltage):
        if np.absolute(outputVoltage) > NI_BNC_MAX_VOLTAGE:
            print "Voltage out of range. Acceptable: {-10:10}"
            return 0
        elif chan not in [0,1]:
            print "Channel %d doesn't exist."%chan
            return 0
        else:

            # Create output data (must be array of dtype=numpy.float64)
            outputArray = np.ones((1000,),dtype = np.float64) * outputVoltage;

            
            try:
                # Create handle for the task.
                taskHandle = TaskHandle();
                DAQmxCreateTask("",byref(taskHandle));

                # Choose voltage output channel (on Dev1 / ao<CHAN>)
                DAQmxCreateAOVoltageChan(taskHandle,"Dev1/ao" + str(chan),"",-10.0,10.0,DAQmx_Val_Volts,"");

                # Open the device
                DAQmxStartTask(taskHandle);
                
                # Write voltage to channel
                DAQmxWriteAnalogF64(taskHandle,1,1,10.0,DAQmx_Val_GroupByChannel,outputArray,None,None);

            # Handle errors if they arise
            except DAQError as err:
                print "DAQmx Error: %s"%err

            # Kill the task
            finally:
                if taskHandle:
                    # Clear the task
                    DAQmxStopTask(taskHandle);
                    DAQmxClearTask(taskHandle);

# Function to zero both outputs
def setZeroOutputs():
    for chan in [0,1]:
        setOutputChanVolt(chan,0)

    
def setSineOutput(chan,amp,freq):
    if chan not in [0,1]:
        print "Chan AO%d not available for output. Try again."%chan
        return 0
    elif np.absolute(amp) > NI_BNC_MAX_VOLTAGE:
        print "Expected wave amplitude %f is out of range."%amp
        return 0
    else:
        
        taskHandle = TaskHandle();
        read = int32();
        
        # Determine length of output data such that full wave is periodically output.
        samples = NI_BNC_MAX_SAMP_RATE / freq;
        wave_out = np.zeros((samples,),dtype = np.float64);
        
        # Generate output wave
        for n in range(samples):
            wave_out[n] = amp * np.sin(2 * np.pi * n * freq / samples)
        
        try:
            # Create handle for the task.
            taskHandle = TaskHandle();
            DAQmxCreateTask("",byref(taskHandle));
            
            # Open output channel "Dev/ao<CHAN>" and set timing.
            DAQmxCreateAOVoltageChan(taskHandle,"Dev1/ao" + str(chan),"",-10.0,10.0,DAQmx_Val_Volts,"");
            #DAQmxCfgSampClkTiming(taskHandle,"",OnboardClock,DAQmx_Val_Rising,DAQmx_Val_ContSamps,samples)
            DAQmxCfgSampClkTiming(taskHandle,"",NI_BNC_MAX_SAMP_RATE,DAQmx_Val_Rising,DAQmx_Val_ContSamps,samples)
        
            # DO WE NEED A WEIRD CALLBACK FUNCTION HERE FOR AFTER SIGNAL GENERATION?
            # Maybe not. Unsure.
            
            # Prepare channel for wave output
            DAQmxWriteAnalogF64(taskHandle,samples,0,10,DAQmx_Val_GroupByChannel,wave_out,None,None);
            
            # Start waveform output
            DAQmxStartTask(taskHandle);
            
        # Handle errors
        except DAQError as err:
            print "DAQmx Error: %s"%err
        
        # Kill and clear the task
        finally:
            print "still running"
            if taskHandle:
            
                DAQmxStopTask(taskHandle);
                DAQmxClearTask(taskHandle);

###################################################
"""
Class for generating a continuous output waveform
    from the NI-BNC box.
    
Input on initialization requires the output 
    channel number and sampling rate to be used.
    (Must be less than NI_BNC_MAX_SAMP_RATE or
    it will just assume the max.)
    
setWave function must be called with the wave as 
    input (of array dtype=numpy.float64). The max
    amplitude is NI_BNC_MAX_VOLTAGE. If the array
    has a larger value than the maximum, the wave
    will be scaled down to satisfy the maximum.
    
    After startWave has been called, setWave may
    be called again to re-write the output wave
    without restarting the entire task.
    
startWave and endWave are self-explanatory, but
    it is important to note that if endWave is 
    not called before program termination, the 
    BNC box will continue to output the last
    setWave.


"""
###################################################
def GetTerminalNameWithDevPrefix(taskhandle, terminalname):
    """Gets terminal name with device prefix.
       Returns trigger name."""
    ndev = GetTaskNumDevices(taskhandle)
    for i in range(1, ndev+1):
         device = GetNthTaskDevice(taskhandle, i, 256)
         pcat = GetDevProductCategory(device)
         if pcat != Val_CSeriesModule and pcat != Val_SCXIModule:
             triggername = "/" + device + "/" + terminalname
             break
    return triggername
    
class NIReadWaves2(Task):
    def __init__(self,chans=None, rate=None, bufferLength=10000):
        Task.__init__(self)
        self.chans = chans
        # trigname =GetTerminalNameWithDevPrefix("", "ai/SampleClock")
        self.callback = None
        self.bufferLength = bufferLength
        self.data = np.zeros((self.bufferLength*len(self.chans),), dtype=np.float64)
        self.data_lists = {}
        n = 0
        for chan in self.chans:
            self.data_lists[n] = np.zeros((self.bufferLength,), dtype=np.float64)
            n += 1
        self.callbackN = 0
        
        # Check that rate doesn't exceed the max. If it does, set to max.
        if rate <= NI_BNC_MAX_SAMP_RATE:
            self.rate = rate
        else:
            self.rate = NI_BNC_MAX_SAMP_RATE

        # Create the input voltage channels.
        for chan in self.chans:
            self.CreateAIVoltageChan("Dev1/ai"+str(chan),"",DAQmx_Val_RSE,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",self.bufferLength,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.bufferLength)
        # self.CfgDigEdgeStartTrig("",trigName,DAQmx_Val_Rising)
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,self.bufferLength/2,0)
        self.AutoRegisterDoneEvent(0)
    
    def EveryNCallback(self):
        self.callbackN += 1
        read = int32()
        self.ReadAnalogF64(self.bufferLength,10.0,DAQmx_Val_GroupByChannel,self.data,self.bufferLength,byref(read),None)
        n=0
        for chan in self.chans:
            start = (self.callbackN%2)/2.*self.bufferLength
            stop = ((self.callbackN%2+1)/2.)*self.bufferLength
            self.data_lists[n][start:stop] = self.data[n*self.bufferLength+start:(n)*self.bufferLength+stop]
            n+=1
        if self.callbackN%2==0: self.callback(self.data_lists)
        return 0 # The function should return an integer
    
    def DoneCallback(self, status):
        #print "Status",status.value
        return 0 # The function should return an integer
    
    def setCallback(self, function):
        self.callback = function

    def startWave(self):
        # Starts the wave output.
        self.StartTask()
    
    def endRead(self):
        # End all output of the wave and clear the task.
        self.StopTask()
        self.ClearTask()


class NIWaves(Task):
    def __init__(self,outchan=None, readchans=None, rate=None, bufferLength=10000):
        Task.__init__(self)
        self.outchan = outchan
        self.readchans = readchans
        self.callback = None
        
        # Check that rate doesn't exceed the max. If it does, set to max.
        if rate <= NI_BNC_MAX_SAMP_RATE:
            self.rate = rate
        else:
            self.rate = NI_BNC_MAX_SAMP_RATE

        # Create the output voltage channel.
        self.CreateAOVoltageChan("Dev1/ao"+str(self.outchan),"",-10.0,10.0,DAQmx_Val_Volts,"")
        # Create the input voltage channels.
        for chan in self.readchans:
            self.CreateAIVoltageChan("Dev1/ai"+str(chan),"",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
        
    def setWave(self,wave=None):
        
        # Check that the input wave is within max/min range of the device. 
        # If not, scale to fit.
        if max(wave) > NI_BNC_MAX_VOLTAGE:
            wave = (wave/max(wave)) * NI_BNC_MAX_VOLTAGE
        if np.absolute(min(wave)) > NI_BNC_MAX_VOLTAGE:
            wave = (wave/np.absolute(min(wave))) * NI_BNC_MAX_VOLTAGE
        
        self.buffLength = len(wave)
        self.CfgSampClkTiming("",self.rate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.buffLength)
        self.WriteAnalogF64(self.buffLength,0,10,DAQmx_Val_GroupByChannel,wave,None,None)
        
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,self.buffLength,0)
        self.AutoRegisterDoneEvent(0)
        
        self.data = np.zeros((self.bufferLength*len(self.readchans),), dtype=np.float64)
    
    def EveryNCallback(self):
        read = int32()
        self.ReadAnalogF64(self.buffLength,10.0,DAQmx_Val_GroupByChannel,self.data,self.buffLength,byref(read),None)
        self.data_lists = {}
        n=0
        for chan in self.readchans:
            self.data_lists[n] = self.output[n*self.buffLength:(n+1)*self.buffLength]
            n+=1
        self.callback(self.data_lists)
        return 0 # The function should return an integer
    
    def DoneCallback(self, status):
        #print "Status",status.value
        return 0 # The function should return an integer
    
    def setCallback(self, function):
        self.callback = function

    def startWave(self):
        # Starts the wave output.
        self.StartTask()
    
    def endWave(self):
        # End all output of the wave and clear the task.
        self.StopTask()
        self.ClearTask()

class NIOutputWave(Task):
    def __init__(self,chan=None,rate=None):
        Task.__init__(self)
        self.chan = chan
        
        # Check that rate doesn't exceed the max. If it does, set to max.
        if rate <= NI_BNC_MAX_SAMP_RATE:
            self.rate = rate
        else:
            self.rate = NI_BNC_MAX_SAMP_RATE

        
        # Create the output voltage channel.
        self.CreateAOVoltageChan("Dev1/ao"+str(self.chan),"",-10.0,10.0,DAQmx_Val_Volts,"")
        
    def setWave(self,wave=None):
        
        # Check that the input wave is within max/min range of the device. 
        # If not, scale to fit.
        if max(wave) > NI_BNC_MAX_VOLTAGE:
            wave = (wave/max(wave)) * NI_BNC_MAX_VOLTAGE
        if np.absolute(min(wave)) > NI_BNC_MAX_VOLTAGE:
            wave = (wave/np.absolute(min(wave))) * NI_BNC_MAX_VOLTAGE
                    
        self.wave = wave
        self.buff = len(wave)
        self.CfgSampClkTiming("",self.rate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.buff)
        self.WriteAnalogF64(self.buff,0,10,DAQmx_Val_GroupByChannel,self.wave,None,None)

    def startWave(self):
        # Starts the wave output.
        self.StartTask()
    
    def endWave(self):
        # End all output of the wave and clear the task.
        self.StopTask()
        self.ClearTask()
        
###################################################
"""
Class for reading data from 1-to-n channels on the
    NI-BNC2110 box.
    
Input on initialization requires the input 
    channel numbers as a list and sampling rate to 
    be used. (Must be less than NI_BNC_MAX_SAMP_RATE 
    or it will just assume the max.)
    
readWave function must be called with the number of 
    samples per channel to be read. This will return
    the data from all channels to the caller as a
    single array which must be disseminated later.

"""
###################################################
        
class NIReadWaves(Task):
    def __init__(self,chans=None,rate=None):
        Task.__init__(self)
        self.chans=chans
        
        # Check that rate doesn't exceed the max. If it does, set to max.
        if rate <= NI_BNC_MAX_SAMP_RATE:
            self.rate = rate
        else:
            self.rate = NI_BNC_MAX_SAMP_RATE
        
        # Create the input voltage channels.
        for chan in self.chans:
            self.CreateAIVoltageChan("Dev1/ai"+str(chan),"",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
        
    def readWaves(self,samples=None):
        
        # Handles the reading of the waves at the specified rate from both channels.
        # Returns data for both channels at once; needs to be handled by the calling
        #     program to be split into X and Y (e.g.).
        
        self.read=int32()
        
        self.samples = samples
        self.buffer = self.samples*len(self.chans)
        self.output = np.zeros((self.buffer,), dtype=np.float64)
        
        self.CfgSampClkTiming("",self.rate,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,self.samples)
        
        self.StartTask()
        self.ReadAnalogF64(self.samples,10.0,DAQmx_Val_GroupByChannel,self.output,self.buffer,byref(self.read),None)
        self.StopTask()
        
        self.data_lists = {}
        n=0
        for chan in self.chans:
            self.data_lists[n] = self.output[n*self.samples:(n+1)*self.samples]
            n+=1
            
        return self.data_lists
        
    def endRead(self):
        self.StopTask()
        self.ClearTask()
        
        
# Function to synchronously read voltages from one-to-n channels
# Input should be a list, even if only one channel to be read.
def readChanVoltages(chans,samples):
       
    # Declaration of variable passed by reference
    taskHandle = TaskHandle()
    read = int32()
    
    # Setup data storage buffer and variable
    buffer = samples * len(chans);
    data = np.zeros((buffer,), dtype=np.float64)

    try:
        # DAQmx Configure Code
        DAQmxCreateTask("",byref(taskHandle))
        
        # Add each input channel to the task.
        for chan in chans:
            # Choose channel for input and timing on channel.
            DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai"+str(chan),"",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
        
        # Clock setup
        DAQmxCfgSampClkTiming(taskHandle,"",NI_BNC_MAX_SAMP_RATE,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,samples)
    
        # DAQmx Start Code
        DAQmxStartTask(taskHandle)
    
        # DAQmx Read Code
        DAQmxReadAnalogF64(taskHandle,samples,10.0,DAQmx_Val_GroupByChannel,data,buffer,byref(read),None)
    
        print "Acquired %d points"%read.value
    except DAQError as err:
        print "DAQmx Error: %s"%err
    finally:
        if taskHandle:
            # DAQmx Stop Code
            DAQmxStopTask(taskHandle)
            DAQmxClearTask(taskHandle)
            return data

# Some example code

# freq = 2
# amp=5
# samples = NI_BNC_MAX_SAMP_RATE / freq;
# wave_out = np.zeros((samples,),dtype = np.float64);

# Generate output wave
# for n in range(samples):
    # wave_out[n] = amp * np.sin(2 * np.pi * n * freq / samples)

# waveOut = NIOutputWave(0,NI_BNC_MAX_SAMP_RATE)
# waveOut.setWave(wave_out)
# waveIn = NIReadWaves([0],10000)
# waveOut.startWave()
# data = waveIn.readWaves(10000)
# waveOut.endWave()
# waveIn.endRead()

