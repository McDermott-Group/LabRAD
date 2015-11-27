# Copyright (C) 2015 Samuel Owen, Ivan Pechenezhskiy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
This module could be used to create the waveforms that are used to
populate the DAC boars. See the __main__ section of this file for
examples.
"""

import numpy as np
import scipy.signal as ss

import labrad.units as units


class _WavePulse():
    """
    Base pulse class that contains shared methods.
    """
    def _ns(time):
        """
        Convert time to nanoseconds. Return an integer without any
        attached units.
        """
        if isinstance(time, units.Value):
            time = time['ns']
        return int(np.round(time))
    
    def _init_times(self, start=None, duration=None, end=None):
        """
        Define the pulse start, end, and duration attributes.
        
        Inputs:
            start: start time of the pulse.
            duration: duration of the pulse.
            end: end time of the pulse.
        """
        if [start, duration, end].count(None) > 1:
            raise ValueError("A pair of time parameters is required " +
                    "to define a pulse. These possible time " +
                    "parameters are 'start', 'duraction', and 'end'.")
                    
        if start is not None:
            self.start = self._ns(start)
        
        if duration is not None:
            self.duration = self._ns(duration)
        
        if end is not None:
            self.end = self._ns(end) 
        
        if start is None:
            self.start = self.end - self.duration + 1

        if duration is None:
            self.duration = self.end - self.start + 1
            
        if end is None:
            self.end = self.start + self.duration - 1
        
        if self.start > self.end:
            raise ValueError("The pulse ends before it starts: " +
                    "the pulse starts at " + str(self.start) + " ns, " +
                    "and it ends at " + str(self.end) + " ns.")
                    
       if self.end - self.start + 1 > self.duration:
            raise ValueError("Inconsistent time parameters: the pulse" +
                    " starts at " + str(self.start) + " ns, its " +
                    "duration is " + str(self.duration) + " ns, and " +
                    "it ends at " + str(self.end) + " ns.")
    
    def _init_amplitude(self, amplitude):
        """
        Define the amplitude (strip units from the amplitude value).
        
        Input:
            amplitude: amplitude of the pulse.
        """
        if isinstance(amplitude, units.Value):
            self.amplitude = amplitude[units.Unit(amplitude)]
        else:
            self.amplitude = float(amplitude)
        if abs(self.amplitude) > 1:
            raise ValueError("The pulse amplitude should not exceed 1.")
        
    def _init_harmonic(frequency, phase):
        """
        Define the pulse frequency and phase.
        
        Inputs:
            frequency: frequency of the harmonic pulse.
            phase: phase of the harmonic pulse.
        """
        if isinstance(frequency, units.Value):
            self.frequency = frequency['GHz']
        else:
            self.frequency = float(frequency)
        
        if isinstance(phase, units.Value):
            self.phase = phase['rad']
        else:
            self.phase = float(phase)
    
    def after(self, time=0):
        """
        Time point after the pulse.
        
        Input:
            time: time delay after this pulse in ns.
        """
        return self.end + 1 + self._ns(time)
        
    def before(self, time=0):
        """
        Time point before the pulse.
        
        Input:
            time: time delay before this pulse in ns.
        """
        return self.start - 1 - self._ns(time)
       
       
class DC(_WavePulse):
    """
    DC pulse.

    Inputs:
        amplitude: amplitude of the dc pulse.
        start: starting time of the dc pulse.
        duration: length of the dc pulse.
        end: ending time of the dc pulse.
    """
    def __init__(self, amplitude=0, start=None, duration=None, end=None):
        self._init_times(start, duration, end)
        self._init_amplitude(amplitude)
        
        self.pulse = np.full(self.duration, self.amplitude)


class Sine(_WavePulse):
    """
    Sine pulse.

    Inputs:
        amplitude: amplitude of the sine pulse.
        frequency: frequency of the sine pulse.
        phase: phase of the sine pusle.
        start: starting time of the sine pulse.
        duration: length of the sine pulse.
        end: ending time of the sine pulse.
    """
    def __init__(self, amplitude=0, frequency=0, phase=0,
            start=None, duration=None, end=None):
        self._init_times(start, duration, end)
        self._init_amplitude(amplitude)
        self._init_harmonic(frequency, phase)

        t = np.linspace(0, self.duration - 1, self.duration)
        self.pulse = (self.amplitude *
                np.sin(2 * np.pi * self.frequency * t + self.phase))

                
class Cosine(_WavePulse):
    """
    Cosine pulse.

    Inputs:
        amplitude: amplitude of the cosine pulse.
        frequency: frequency of the cosine pulse.
        phase: phase of the cosine pusle.
        start: starting time of the cosine pulse.
        duration: length of the cosine pulse.
        end: ending time of the cosine pulse.
    """
    def __init__(self, amplitude=0, frequency=0, phase=0,
            start=None, duration=None, end=None):
        self._init_times(start, duration, end)
        self._init_amplitude(amplitude)
        self._init_harmonic(frequency, phase)

        t = np.linspace(0, self.duration - 1, self.duration)
        self.pulse = (self.amplitude *
                np.cos(2 * np.pi * self.frequency * t + self.phase))


class Gaussian(_WavePulse):
    """
    Gaussian window pulse. The pulse is trunctated at about 1 per 2^14
    level since the DACs have 14-bit resolution.

    Inputs:
        amplitude: amplitude of the gaussian pulse.
        start: starting time of the gaussian pulse.
        duration: length of the gaussian pulse.
        end: ending time of the gaussian pulse.
    """
    def __init__(self, amplitude=0, start=None, duration=None, end=None):
        self._init_times(start, duration, end)
        self._init_amplitude(amplitude)

        sigma = (float(self.duration) - 1) / np.sqrt(112 * np.log(2))
        self.pulse = self.amplitude * ss.gaussian(self.duration, sigma)


class WaveForm():
    """
    Create a waveform from pulses.
    
    The start of one pulse is expected to be one unit
    (i.e. one nanosecond) after the end of the previous pulse
    (i.e. pulse2.end - pulse1.start >= 1). Therefore, to make pulse B
    start immediately after another pulse A initialize B.start to
    (A.end + 1), or simply assign A.after() to B.start.
    
    Input:
        args: arbitrarily long set of _WavePulses to create the waveform
            from. To create a _WavePulse use one of the "public"
            classes such as DC, Sine, Cosine, etc.
    """
    def __init__(self, label='None', *args):
        self.label = label
        pulses = []
        for arg in args:
            if isinstance(arg, _WavePulse):
                pulses.append(arg)

        if len(pulses) > 0:
            # Sort based on the start times.
            for i in range(len(pulses))[::-1]:
                for j in range(i):
                    if pulses[j].start > pulses[j + 1].start:
                        tmp = pulses[j + 1]
                        pulses[j + 1] = pulses[j]
                        pulses[j] = tmp
                
            # Ensure there are no overlaps.
            for i in range(len(pulses) - 1):
                if pulses[i].end > pulses[i + 1].start:
                    raise ValueError("There are overlaps between " + 
                            "the waveform pulses.")
                
            # Loop through and fill unused spots with zeros.
            pulses_filled = []
            for i in range(len(pulses) - 1):
                pulses_filled.append(pulses[i].pulse)
                gap = pulses[i + 1].start - pulses[i].end
                if gap > 1:
                    pulses_filled.append(np.zeros(gap - 1))   
            pulses_filled.append(pulses[len(pulses) - 1].pulse)
            self.pulses = np.hstack(pulses_filled)
        else:
            self.pulses = np.array([0])
            
    self.start = pulses[0].start
    self.end = pulses[-1].end
    self.duration = self.end - self.start + 1

            
def wfs2dict(min_length=20, *args):
    """
    Return a waveform dictionary with the waveform labels as the keys.
    
    Align the waveforms using the waveform starting time. Ensure that
    the waveforms are of an equal length and that they are longer than
    the minimum length.
    
    Input:
        min_length: mininum allowed length of the final wavefrom. Short
            waveforms will be padded with zeros.
        args: arbitrarily long set of WaveForms.
        
    Output:
        waveforms: dictionary with the processed waveforms.
    """
    wfs = []
    for arg in args:
        if isinstance(arg, _WavePulse):
            wfs.append(arg)
    
    # Align the waveforms and append a zero to the start and end.
    start = min([wf.start for wf in wfs]) - 1
    for i in range(len(wfs)):
        wfs[i].pulses = np.hstack([np.zeros(wfs[i].start - start),
                                   wf[i].pulses, 0])
     
    # Ensure that the waveforms are long enough and are of an equal
    # length.
    length = max([wf.pulses.size for wf in wfs])
    start = max(0, (min_length - length) / 2)
    for i in range(len(wfs)):
        end = length - start - wfs[i].pulses.size
        wfs[i].pulses = np.hstack([np.zeros(start), wfs[i].pulses,
                                   np.zeros(end)])
        
    return {wf.label: wf.pulses for wf in wfs}


if __name__ == "__main__":
    """
    Tests and examples. Feel free to add your test/example at the end.
    """
    # Cosine pulse with amplitude of 1 and frequency of 0.25 GHz
    # starting at t = 2 ns and ending at t = 8 ns.
    pulseA1 = Cosine(amplitude=1, frequency=0.25, start=2, end=8)

    # Sine pulse with amplitude of 0.5 and frequency of 0.25 GHz
    # starting at the start of pulseA1 and ending at the end of pulseA1.
    pulseB1 = Sine(amplitude=0.5, frequency=0.25,
                   start=pulseA1.start, end=pulseA1.end)
  
    # DC pulse with amplitude of -1 starting after the end of pulseA1.
    # The pulse duration is 10 ns.
    pulseB2 = DC(amplitude=-1, start=pulseA1.after(), duration=10)
    
    # Combine the two pulses into one waveform. The waveform class
    # automatically puts the wave pulses in the correct order.
    waveformB = WaveForm('B', pulseB1, pulseB2)
    
    # Specifing the start, duration and end times at the same time will
    # work only if these parameters are consistent, i.e. if the equation
    # self.duration = self.end - self.start + 1 is satisfied.
    pulseA2 = DC(start=pulseB2.after(-1), duration=11, end=pulseB2.end)
    try:
        # Incosistent specifications.
        pulseA2 = DC(start=pulseB2.after(-1), duration=12, end=pulseB2.end)
    except ValueError
        print('The error has been correctly caught.')
    
    # Sine pulse with amplitude of 1 and frequency of 0.1 GHz
    # starting 2 ns after pulseB2 and ending at the same time as 
    # pulseB2.
    pulseA2 = Sine(amplitude=1, phase=np.pi/2, frequency=0.1,
                   start=pulseB2.after(2), end=pulseB2.end)
    
    # Combine the two pulses into one waveform. The waveform class
    # automatically puts the wave pulses in the correct order.
    waveformA = WaveForm('A', pulseA1, pulseA2)
    
    # Create a waveform dictionary with the waveform labels as the keys.
    # The waveforms will be aligned based on their start times. They
    # will be zero-padded to ensure equal length that is longer
    # a minimum length (min_lenght=20, by default)
    waveforms = wfs2dict(waveformA, waveformB)
    print(waveforms)

    # Gaussian pulse with amplitude of 1 and frequency of 0.25 GHz
    # starting at t = 0 ns and ending at t = 14 ns (duration is equal
    # 15 ns).
    pulseC = Gaussian(amplitude=1, start=0, duration=15, end=pulseB2.end)
    waveformC = WaveForm('C', pulseC)
    
    waveforms = wfs2dict(waveformA, waveformB, waveformC)
    print(waveforms)