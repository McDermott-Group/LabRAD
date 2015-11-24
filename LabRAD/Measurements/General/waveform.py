import wavepulse as wp
import numpy as np

class WaveForm():
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
    
    def __init__(self, label, *argv):
        waves = []
        self.label = label
        for arg in argv:
            if isinstance(arg, wp.WavePulse):
                waves.append(arg)
            
        # sort based on start times
        for i in range(len(waves))[::-1]:
            for j in range(i):
                if (waves[j].start > waves[j + 1].start):
                    tmp = waves[j + 1]
                    waves[j + 1] = waves[j]
                    waves[j] = tmp
            
        # ensure there are no overlaps     
        for i in range(len(waves) - 1):
            if(waves[i].end >= waves[i + 1].start):
                raise ValueError('Invalid Pulses: There are overlaps')
            
        # loop through and fill unused spots with 0's
        wavesFilled = []
        
        for i in range(len(waves) - 1):
            gap = waves[i + 1].start - waves[i].end
            if gap > 1:
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
    wave1 = WaveForm('First',wp.WavePulse(type="block", start=t_0, amplitude=3, frequency=2, end=t_1),wp.WavePulse(type="sine", start=t_2 ,amplitude=10, frequency=0.25, duration=10))
    print wave1.label
    print wave1.getArr()
    print '\n'
    
    pulseA1 = wp.WavePulse(type="cosine", start=t_0, amplitude=5, frequency=0.125, end=t_1)

    pulseB1 = wp.WavePulse(type="sine", start=pulseA1.start, amplitude=5, frequency=0.125, end=t_1)
    # Start is specified at an offset to the first pulse in another wave.
    pulseB2 = wp.WavePulse(type="block", start=pulseA1.after(0), amplitude=-2, frequency=0, end=t_2, duration=10)
    waveB = WaveForm('B', pulseB1, pulseB2)
    
    # Bug: conflicting start, end and duration should throw up an error.
    pulseA2 = wp.WavePulse(type="block", start=pulseB2.after(-1), amplitude=1, frequency=0, duration=15, end=pulseB2.end)
    waveA = WaveForm('A', pulseA1, pulseA2)
    
    # Nice to have feature:
    # It would be great to have a way to fully define waveform waveA before
    # waveB. Right now, pulse B2 should be specified before it could be
    # used to assign the start time for pulse A2.
    
    print waveA.label
    print waveA.getArr()
    
    print waveB.label
    print waveB.getArr()
    
    # Bug: phase is not a recognized parameter.
    pulseC = wp.WavePulse(type="block", start=pulseB2.after(-1), phase=1.67, amplitude=1, duration=15, end=pulseB2.end)
    waveC = WaveForm('C', pulseC)
    
    # Semi-bug: block pulse should not require frequency and phase parameters.
    pulseC = wp.WavePulse(type="block", start=pulseB2.after(-1), amplitude=1, duration=15, end=pulseB2.end)
    waveC = WaveForm('C', pulseC)
    
    # Suggestions:
    # 1. Merge wavepulse.py and wavepulse.py. These are two simple, logically
    # related classes that won't be used one without another.
    # 2. Rename 'block' to 'dc'.
    # 3. Make the pulse names case insensitive, i.e.
    # 'Sine' or 'sine' should be both valid options (and maybe even 'sin'?).
    
    # Next step:
    # Write a method or a function that takes all given waveforms, and ensure
    # that they are of an equal length by padding the waveforms with zeros on both sides.
    # Here is the list of the ultimate requirements that all waveforms should satisfy
    # after postprocessing:
    # 1. Each waveform should automatically get at least one zero (0) 
    # appended to its start and end.
    # 2. Each waveform should be longer than 20.
    # 3. All waveforms should be of an equal length.
    