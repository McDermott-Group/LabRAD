import niPCI6221 as ni
import numpy as np
import time
import matplotlib.pyplot as plt

def CallMe(data):
    print "CallMe"
    #plt.plot(data)
    #plt.ylabel('time averaged')
    #plt.show()
    


readBuf = np.array([])
t = np.linspace(0, 2*np.pi, 1000)
writeBuf = 2*np.sin(t)
writeBuf.astype(np.float64)

inp = ni.CallbackTask()
inp.configureCallbackTask("Dev1/ai0", 1000.0, 1000)
inp.setCallback(CallMe)


out = ni.AnalogOuputTask()
out.configureAnalogOutputTask("Dev1/ao0", 1000.0, writeBuf)

out.StartTask()
inp.StartTask()

raw_input('Acquiring samples continuously. Press Enter to interrupt\n')

inp.StopTask()
inp.ClearTask()
out.StopTask()
out.ClearTask()



