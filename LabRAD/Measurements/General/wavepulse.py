#import pulse_shapes as pulse
import numpy as np

#######################################################################
############## HOW TO USE WAVEFORM CLASS ##############################
#
# import path.to.wavepulse as wp
# import path.to.waveform as wf
#
# r = wp.WavePulse("block", 0 ,3, 2, 10, None)
# s = wp.WavePulse("sine", 0, 10, 0.25, None, 10)
# w = wf.WaveForm(r,s, ...) #as many as you want
#
# w.getArr() gives you the array that you need
# r.start, r.end and r.duration return exactly what you think
#
# wavepulses are declared with (type, start, amplitude, frequency, end=None, duration=None)
# where type is either "block", "sine", "cosine", or "gauss"
#######################################################################
#######################################################################

class WavePulse ():
    def __init__(self, type, start, amplitude, frequency, end=None, duration=None):
        #type either block, sine, cosine or gauss
        self.type = type
        self.start = start
        self.amplitude = amplitude
        self.frequency = frequency
    
        if end is None:
            if duration is None:
                duration = 0
                self.duration = 0
            #elsen
            #    self.duration = duration
            self.end = start + duration
        else:
            self.end = end
            
        if duration is None:
            self.duration = max(0,end - start) 
        else:
            self.duration = duration
        self.wave = {
            "block" : self.block,
            "sine"  : self.sine,
            "cosine": self.cosine,
            "gauss" : self.gauss
        }

    def toArray(self):
        return self.wave[self.type]()
        
    def block(self):
        return np.zeros(int(self.duration)) + self.amplitude
        
    def sine(self):
        t = np.linspace(0, int(self.duration) - 1, int(self.duration))
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        
    def cosine(self):
        t = np.linspace(0, int(self.duration) - 1, int(self.duration))
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t)
        
    def gauss(self):
        '''
        Returns a "slowed" square pulse that consists of gaussian rise and
        fall times and a DC segment. Length is the length of the DC part of
        the pulse in ns, and FW is the width at 1/10 maximum of the gaussian
        rise and fall. Total pulse length is thus ~= length + 2 * FW.
        '''
        FW = self.frequency #unsure what FW should be
        
        c = FW / (2. * np.sqrt(2. * np.log(10.)))
        t = np.linspace(0, int(self.duration + 3. * FW) - 1, int(self.duration + 3. * FW))
        pls = np.zeros(len(t))
        t1 = t < FW
        t2 = (t >= FW) & (t < FW + self.duration)
        t3 = (t >= FW + self.duration)
        
        pls[t1] = self.amplitude * np.exp(-np.power(t[t1] - FW, 2.) / 2. / c**2.)
        pls[t2] = self.amplitude
        pls[t3] = self.amplitude * np.exp(-np.power(t[t3] - FW - self.duration, 2.) / 2. / c**2.)
        return pls
        