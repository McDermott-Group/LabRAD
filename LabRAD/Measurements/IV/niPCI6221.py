try:
    from PyDAQmx import *
except ImportError:
    print "This program requires PyDAQmx to run."
from PyDAQmx.DAQmxCallBack import *
import numpy as np
import ctypes
import re

NI_PCI_6221_MAX_VOLTAGE = 10.0;  # Max voltage output
NI_PCI_6221_MAX_SAMP_RATE = 10000;   # Measurement/generation rate ?? double check this

class NIPCI6221(object):
    """ This Class is a set of methods for controlling the NI PCI 6221 boards."""
    def __init__(self):
        pass

    def configureSynchronizedAIAOTask(self, analogInput, acAnalogOutput, dcAnalogOutput, sampRate, readBuffer, writeBuffer, dcValue):
        try:
            aiTaskHandle = TaskHandle()
            acTaskHandle = TaskHandle()
            dcTaskHandle = TaskHandle()

            # DAQmx Configure Analog Input
            if re.match("Dev1/ai\d", element):
                DAQmxCreateTask("",byref(aiTaskHandle))
                DAQmxCreateAIVoltageChan(aiTaskHandle,"Dev1/ai1","",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
                DAQmxCfgSampClkTiming(aiTaskHandle,"",10000.0,DAQmx_Val_Rising,DAQmx_Val_ContSamps,buffsize)
            else:
                print "Analog inputs by the name of ", analogInput, " are not supported by this instrument"

            trigName = self.GetTerminalNameWithDevPrefix(aiTaskHandle, "ai/StartTrigger")
            print "trigName=", trigName

            #DAQmx Configure AC Analog Output
            if re.match("Dev1/ao\d", element):
                DAQmxCreateTask("",byref(acTaskHandle))
                DAQmxCreateAOVoltageChan(acTaskHandle,"Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,None)
                DAQmxCfgSampClkTiming(acTaskHandle, "", sr/2,DAQmx_Val_Rising,DAQmx_Val_ContSamps,buffsize)
                DAQmxCfgDigEdgeStartTrig(acTaskHandle,trigName,DAQmx_Val_Rising)
            else:
                print "Analog outputs by the name of ", analogOutput, " are not supported by this instrument"

            DAQmxStopTask(aiTaskHandle)
            DAQmxClearTask(aiTaskHandle)
            DAQmxStopTask(acTaskHandle)
            DAQmxClearTask(acTaskHandle)
        except DAQError as err:
            print "DAQmx Error: %s"%err

    def GetTerminalNameWithDevPrefix(self, taskHandle, terminalName):
        device = ctypes.create_string_buffer(256)
        productCategory = int32()
	numDevices = uInt32()
	i = 1
	try:
            DAQmxGetTaskNumDevices(taskHandle,byref(numDevices))
            for i in range(1, numDevices.value+1):
                DAQmxGetNthTaskDevice(taskHandle,i,device,256)
                DAQmxGetDevProductCategory(device,byref(productCategory))
                if productCategory.value!=DAQmx_Val_CSeriesModule and productCategory.value!=DAQmx_Val_SCXIModule:
                    triggername = "/" + device.value + "/" + terminalName
                    break
        except DAQError as err:
            print "DAQmx Error: %s"%err
        return triggername
