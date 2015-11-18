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

trigName = ""

class CallbackTask(Task):
    def __init__(self):
        Task.__init__(self)
        self.buffLen = 0
        self.callback =None
        
    def configureCallbackTask(self, analogInputNameStr, sampRate, numSamples):
        self.buffLen = numSamples
        self.data = np.zeros(self.buffLen)
        self.CreateAIVoltageChan(analogInputNameStr,"",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("",float(sampRate),DAQmx_Val_Rising,DAQmx_Val_ContSamps,numSamples)
        trigName = self.GetTerminalNameWithDevPrefix("ai/StartTrigger")
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

class AnalogOuputTask(Task):
    def __init__(self):
        Task.__init__(self)
        self.buffLen = 0
        
    def configureAnalogOutputTask(self, analogOutputNameStr, outputRate, outWaveForm):
        self.buffLen = len(outWaveForm)
        self.CreateAOVoltageChan(analogOutputNameStr,"",-10.0,10.0,DAQmx_Val_Volts,None)
        self.CfgSampClkTiming("", outputRate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,self.buffLen)
        trigName = "/Dev1/ai/StartTrigger" # pass from one to the other !!!!!!!!!!!!
        self.CfgDigEdgeStartTrig(trigName,DAQmx_Val_Rising)
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
