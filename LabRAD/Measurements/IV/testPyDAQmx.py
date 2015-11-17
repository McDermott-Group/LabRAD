try:
    from PyDAQmx import *
except ImportError:
    print "This program requires PyDAQmx to run."
from PyDAQmx.DAQmxCallBack import *
import numpy as np
import ctypes 


class MyList(list):
    pass

buffsize = 100

AItaskHandle = TaskHandle()
numSampsRead = int32()
data = np.zeros((buffsize,),dtype=np.float64)

numperiods = 3
period = 2.0
sr = buffsize/period
t = np.linspace(0, 4*np.pi, buffsize)
dataw = 2*np.sin(t)
dataw.astype(np.float64)

AOtaskHandle = TaskHandle()

data1 = np.array([]) #MyList()
id_data = create_callbackdata_id(data1)

def GetTerminalNameWithDevPrefix(taskHandle, terminalName):
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

def EveryNCallback_py(taskHandle, everyNsamplesEventType, nSamples, callbackData_ptr):
                          
    callbackdata = get_callbackdata_from_id(callbackData_ptr)
    numReadBack = int32()
    data1=np.zeros(100)
    DAQmxReadAnalogF64(taskHandle,buffsize,10.0,DAQmx_Val_GroupByChannel,data1,buffsize,byref(numReadBack),None)                  
    print "Acquired %d samples"%numReadBack.value
    return 0 # The function should return an integer

def DoneCallback_py(taskHandle, status, callbackData_ptr):
    print "Status", status.value
    return 0

def setCallback(self, fn):
    self.everyNcallback=fn

#self.everyNcallback(data)

#def configureSynchronizedAIAOTask(analogInput, acAnalogOutput, dcAnalogOutput, sampRate, readBuffer, writeBuffer, dcValue)
#def startSynchronizedAIAOTask(AItaskHandle, AOtaskHandle, DCtaskHandle)
#def stopSynchronizedAIAOTask(AItaskHandle, AOtaskHandle, DCtaskHandle)
    

try:
    # DAQmx Configure Analog Input
    DAQmxCreateTask("",byref(AItaskHandle))
    DAQmxCreateAIVoltageChan(AItaskHandle,"Dev1/ai1","",DAQmx_Val_Diff,-10.0,10.0,DAQmx_Val_Volts,None)
    DAQmxCfgSampClkTiming(AItaskHandle,"",10000.0,DAQmx_Val_Rising,DAQmx_Val_ContSamps,buffsize)
    
    trigName = GetTerminalNameWithDevPrefix(AItaskHandle, "ai/StartTrigger")

    print "trigName=", trigName
    #DAQmx Configure Analog Output 
    DAQmxCreateTask("",byref(AOtaskHandle))
    DAQmxCreateAOVoltageChan(AOtaskHandle,"Dev1/ao0","",-10.0,10.0,DAQmx_Val_Volts,None)
    DAQmxCfgSampClkTiming(AOtaskHandle, "", sr/2,DAQmx_Val_Rising,DAQmx_Val_ContSamps,buffsize)
    DAQmxCfgDigEdgeStartTrig(AOtaskHandle,trigName,DAQmx_Val_Rising)

    EveryNCallback = DAQmxEveryNSamplesEventCallbackPtr(EveryNCallback_py)
    DoneCallback = DAQmxDoneEventCallbackPtr(DoneCallback_py)

    DAQmxRegisterEveryNSamplesEvent(AItaskHandle,DAQmx_Val_Acquired_Into_Buffer,buffsize,0,EveryNCallback,id_data)
    DAQmxRegisterDoneEvent(AItaskHandle,0,DoneCallback,None)  #only applies to finite acquistion

    sampsPerChanWritten = int32()
    DAQmxWriteAnalogF64(AOtaskHandle, buffsize, False, 10.0, DAQmx_Val_GroupByChannel, dataw, byref(sampsPerChanWritten), None)

    DAQmxStartTask(AOtaskHandle)
    DAQmxStartTask(AItaskHandle)

except DAQError as err:
    print "DAQmx Error: %s"%err
finally:
    if AItaskHandle:
        print "Waiting"
        raw_input('Acquiring samples continuously. Press Enter to interrupt\n')
        DAQmxStopTask(AItaskHandle)
        DAQmxClearTask(AItaskHandle)
        DAQmxStopTask(AOtaskHandle)
        DAQmxClearTask(AOtaskHandle)



    
