import wavepulse as wp
import numpy as np

class WaveForm ():
    """
    Create a wave form
    
    label   --  user-defined label for wave form
    argv    --  arbitrarily long list of WavePulses to pass to the waveform
    """
    ############################################################
    ####### MORE DETAILED DOCUMENTATION IN wavepulse.py ########
    ############################################################
    # initializing the wave form takes the most time
    # so do it outside of the run once method
    # 
    # Make sure the start of one pulse is one unit after the
    # end of the previous one (i.e. pulse2.end - pulse1.start >= 1)
    # therefore to make one pulse (B) start immediately after
    # another pulse (A) initialize B.start to (A.end + 1)
    
    def __init__(self,label, *argv):
        waves = []
        self.label = label
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
        """Returns the Waveform as a discrete array"""
        return self.pulses
        
if __name__ == "__main__":
    """
    Creates a sample waveform with a single pulse for 5 ms, a 5ms rest
    and a sine wave of amplitude 10 and frequency of 0.25
    
    Then Createa a cosine wave followed by a block wave with an offset
    specified by the end of the first
    """
    t_0 = 0
    t_1 = 5
    t_2 = 11
    wave1 = WaveForm('First',wp.WavePulse(type="block", start=t_0 ,amplitude=3, frequency=2, end=t_1),wp.WavePulse(type="sine", start=t_2 ,amplitude=10, frequency=0.25, duration=10))
    print wave1.label
    print wave1.getArr()
    print '\n'
    
    pulse1 = wp.WavePulse(type="cosine", start=t_0 ,amplitude=5, frequency=0.125, end=t_1)
    #start is specified at an offset to the first pulse
    pulse2 = wp.WavePulse(type="block", start=pulse1.after(0), amplitude=-2, frequency=0, end=t_2)
    wave2 = WaveForm('second',pulse1,pulse2)
    print wave2.label
    print wave2.getArr()
    
    