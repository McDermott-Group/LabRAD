# TODO:
# DC start trigger

try:
    from PyDAQmx import *
except ImportError:
    print "This program requires PyDAQmx to run."
from PyDAQmx.DAQmxCallBack import *
import numpy as np
import ctypes
import re
import time

NI_PCI_6221_MAX_VOLTAGE = 10.0;  # Max voltage output
NI_PCI_6221_MAX_SAMP_RATE = 10000;   # Measurement/generation rate ?? double check this

#Input 250 kS/s
#Output 833 kS/s for single channel, 740 kS.s for two channels
trigName = ""

class CallbackTask(Task):
    def __init__(self):
        Task.__init__(self)
        self.buffLen = 0
        self.callback =None
        self.trigName=""
        
    def configureCallbackTask(self, analogInputNameStr, sampRate, numSamples):
        self.buffLen = numSamples
        self.data = np.zeros(self.buffLen)
        self.CreateAIVoltageChan(analogInputNameStr,"",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",float(sampRate),DAQmx_Val_Rising,DAQmx_Val_ContSamps,numSamples)
        self.trigName = self.GetTerminalNameWithDevPrefix("ai/StartTrigger")
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,1000,0)
        self.AutoRegisterDoneEvent(0)
        
    def EveryNCallback(self):
        numReadBack = int32()
        self.data=np.zeros(self.buffLen)
        self.ReadAnalogF64(self.buffLen,10.0,DAQmx_Val_GroupByChannel,self.data,self.buffLen,byref(numReadBack),None)
        if self.callback is not None:
            self.callback(self.data)
        return 0 # The function should return an integer
    
    def DoneCallback(self, status):
        print "Status",status.value
        return 0 # The function should return an integer

    def setCallback(self, function):
        self.callback = function
    
    def GetTerminalNameWithDevPrefix(self, terminalName):
        device = ctypes.create_string_buffer(256)
        productCategory = int32()
	numDevices = uInt32()
	i = 1
        self.GetTaskNumDevices(byref(numDevices))
        for i in range(1, numDevices.value+1):
            self.GetNthTaskDevice(i,device,256)
            DAQmxGetDevProductCategory(device,byref(productCategory))
            if productCategory.value!=DAQmx_Val_CSeriesModule and productCategory.value!=DAQmx_Val_SCXIModule:
                triggername = "/" + device.value + "/" + terminalName
                break
        return triggername
    
    def getTrigName(self):
        return self.trigName

class analogInputTask(Task):
    def __init__(self , analogInputNameStr, sampRate, numSamples):
        Task.__init__(self)
        self.buffLen = 0
        self.callback =None
        self.trigName=""
        self.analogInputNameStr = analogInputNameStr
        self.sampRate = sampRate
        self.numSamples = numSamples
        #self.configureCallbackTask()
        
    def configureCallbackTask(self):
        self.read = int32()
        self.buffLen = self.numSamples
        self.data = np.zeros(self.buffLen, dtype=numpy.float64)
        self.CreateAIVoltageChan(self.analogInputNameStr,"",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",float(self.sampRate),DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,self.numSamples)
        self.StartTask()
        self.ReadAnalogF64(self.numSamples,10.0,DAQmx_Val_GroupByChannel,self.data,self.buffLen,byref(self.read),None)
        self.StopTask()
        self.ClearTask()
        return self.data

class acAnalogOutputTask(Task):
    def __init__(self):
        Task.__init__(self)
        self.buffLen = 0
        self.trigName=None
        
    def configureAcAnalogOutputTask(self, acAnalogOutputNameStr, outputRate, outWaveForm, trigName=None):
        self.buffLen = len(outWaveForm)
        self.CreateAOVoltageChan(acAnalogOutputNameStr,"",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("", outputRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.buffLen)
        if trigName is not None:
            self.trigName= trigName
            self.CfgDigEdgeStartTrig(self.trigName,DAQmx_Val_Rising)
        sampsPerChanWritten = int32()
        self.WriteAnalogF64(self.buffLen, False, 10.0, DAQmx_Val_GroupByChannel, outWaveForm, byref(sampsPerChanWritten), None)



    def GetTerminalNameWithDevPrefix(self, terminalName):
        device = ctypes.create_string_buffer(256)
        productCategory = int32()
	numDevices = uInt32()
	i = 1
        self.GetTaskNumDevices(byref(numDevices))
        for i in range(1, numDevices.value+1):
            self.GetNthTaskDevice(i,device,256)
            DAQmxGetDevProductCategory(device,byref(productCategory))
            if productCategory.value!=DAQmx_Val_CSeriesModule and productCategory.value!=DAQmx_Val_SCXIModule:
                triggername = "/" + device.value + "/" + terminalName
                break
        return triggername

class dcAnalogOutputTask(Task):
    def __init__(self):
        Task.__init__(self)
        self.buffLen = 0
        self.trigName=None

    def configureDcAnalogOutputTask(self, dcAnalogOutputNameStr, dcValue):
        self.data=np.zeros(1)
        self.data[0]=dcValue
        self.data.astype(np.float64) #cast to fp64
        self.CreateAOVoltageChan(dcAnalogOutputNameStr,"",-10.0,10.0,DAQmx_Val_Volts,None)
        sampsPerChanWritten = int32()
        self.WriteAnalogF64(1,True,10.0,DAQmx_Val_GroupByChannel,self.data,byref(sampsPerChanWritten),None)
