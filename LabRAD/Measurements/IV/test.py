import niPCI6221 as ni
import numpy as np
import time
import matplotlib.pyplot as plt

def CallMe(data):
    print "CallMe"
    #plt.plot(data)
    #plt.ylabel('time averaged')
    #plt.show()
    
#readBuf = np.array([])
t = np.linspace(0, 2*np.pi, 250000)
writeBuf = 2*np.sin(t)
#writeBuf.astype(np.float64)

inp = ni.CallbackTask() # always start last
inp.configureCallbackTask("Dev1/ai0", 250000.0, 250000)
inp.setCallback(CallMe)
triggerName= inp.getTrigName()
print "trigName=", triggerName

dc=ni.dcAnalogOutputTask()
dc.configureDcAnalogOutputTask("Dev1/ao1",-3.45566)

out = ni.acAnalogOutputTask()
out.configureAcAnalogOutputTask("Dev1/ao0", 250000.0, writeBuf,trigName=triggerName)

dc.StartTask()
out.StartTask()
inp.StartTask()

raw_input('Acquiring samples continuously. Press Enter to interrupt\n')

inp.StopTask()
inp.ClearTask()
out.StopTask()
out.ClearTask()

dc.StopTask()
dc.ClearTask()



