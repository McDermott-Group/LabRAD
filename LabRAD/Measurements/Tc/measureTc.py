import numpy as np
import niPCI6221 as ni

OUTPUT_V = 0.1
BN_PORT = 0
NbSe2_PORT = 1
AVERAGES = 20

class MeasureTc():
    def __init__(self):
        self.initializeDCWave()
        self.initializeReadWaves()
        
    def initializeDCWave(self):
        try:
            self.DCOutput_BN = ni.dcAnalogOutputTask()
            self.DCOutput_BN.configureDcAnalogOutputTask("Dev1/ao"+str(BN_PORT),OUTPUT_V)
            self.DCOutput_BN.StartTask()
            
            self.DCOutput_NbSe2 = ni.dcAnalogOutputTask()
            self.DCOutput_NbSe2.configureDcAnalogOutputTask("Dev1/ao"+str(NbSe2_PORT),OUTPUT_V)
            self.DCOutput_NbSe2.StartTask()
            
            print "started DC outputs"
            
        except Exception as e:
            print 'Error initializing DC outputs:\n' + str(e)
            
    def initializeReadWaves(self):
        try:
            self.waveInput_BN = ni.CallbackTask()
            self.waveInput_BN.configureCallbackTask("Dev1/ai"+str(BN_PORT), AVERAGES,AVERAGES)
            self.waveInput_BN.setCallback(self.updateData_BN)
            self.waveInput_BN.StartTask()
            
            self.waveInput_NbSe2 = ni.CallbackTask()
            self.waveInput_NbSe2.configureCallbackTask("Dev1/ai"+str(NbSe2_PORT), AVERAGES,AVERAGES)
            self.waveInput_NbSe2.setCallback(self.updateData_NbSe2)
            self.waveInput_NbSe2.StartTask()
            
            print "started AC waves"
            
        except Exception as e:
            print 'Error initializing wave output:\n' + str(e)
            
    def updateData_BN(self, data):
        print np.mean(data)
            
    def updateData_NbSe2(self, data):
        print np.mean(data)
        
if __name__ == "__main__":
    app = MeasureTc()