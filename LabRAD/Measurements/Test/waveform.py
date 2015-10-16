import wavepulse as wp
import numpy as np

class WaveForm ():
    # initializing the wave form takes the most time
    # so do it outside of the run once method
    # 
    # Make sure the start of one pulse is one unit after the
    # end of the previous one (i.e. pulse2.end - pulse1.start >= 1)
    # therefore to make one pulse (B) start immediately after
    # another pulse (A) initialize B.start to (A.end + 1)
    
    def __init__(self, *argv):
        waves = []
        for arg in argv:
            if isinstance(arg, wp.WavePulse):
                waves.append(arg)
            
        #sort based on start times
        for i in range(len(waves))[::-1]:
            for j in range(i):
                if (waves[j].start > waves[j + 1].start) :
                    tmp = waves[j + 1]
                    waves[j + 1] = waves[j]
                    waves[j] = tmp
            
        #ensure there are no overlaps     
        for i in range(len(waves) - 1):
            if(waves[i].end >= waves[i + 1].start):
                raise ValueError('Invalid Pulses: There are overlaps')
            
        #loop through and fill unused spots with 0's
        wavesFilled = []
        
        for i in range(len(waves) - 1):
            gap = waves[i + 1].start - waves[i].end
            if (gap > 1) :
                wavesFilled.append(waves[i].toArray())
                wavesFilled.append(np.zeros(int(gap - 1)))
            else:
                wavesFilled.append(waves[i].toArray())
                
        wavesFilled.append(waves[len(waves) - 1].toArray())

        self.pulses = np.hstack(wavesFilled)
        
    def getArr(self):
        return self.pulses