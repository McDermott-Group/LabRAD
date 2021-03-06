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

import collections
import itertools
import warnings
import numpy as np
import scipy.signal as ss

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

from win32api import SetConsoleCtrlHandler

import labrad.units as units

def _flatten(iterable):
    """
    De-nest a list of _WavePulses for convenience.
    
    Input:
        iterable: an iterable object.
    Output:
        list: de-nested list of _WavePulses.
    """
    remainder = iter(iterable)
    while True:
        first = next(remainder)
        if (isinstance(first, collections.Iterable) and
                not isinstance(first, _WavePulse)):
            remainder = itertools.chain(first, remainder)
        else:
            yield first

class _WavePulse():
    """
    Base pulse class that contains shared methods.
    """
    def _ns(self, time):
        """
        Convert time to nanoseconds. Return an integer without any
        units attached.
        
        Input:
            time: physical or numerical (in ns) time value.
        Output:
            time: numerical time value in ns.
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
        Output:
            None.
        """
        if [start, duration, end].count(None) > 1:
            raise ValueError("A pair of time parameters is required " +
                    "to define a pulse. These possible time " +
                    "parameters are 'start', 'duration', and 'end'.")
                    
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
        
        if self.start > self.end + 1:
            raise ValueError("The pulse ends before it starts: " +
                    "the pulse starts at " + str(self.start) + " ns " +
                    "and ends at " + str(self.end) + " ns.")
                    
        if self.end - self.start + 1 != self.duration:
            raise ValueError("Inconsistent time parameters: the pulse" +
                    " starts at " + str(self.start) + " ns, its " +
                    "duration is " + str(self.duration) + " ns, while" +
                    " the pulse is expected to end at " +
                    str(self.end) + " ns.")
    
    def _amplitude(self, amplitude):
        """
        Process the amplitude (strip units from the amplitude value).
        
        Input:
            amplitude: amplitude of the pulse.
        Output:
            amplitude: amplitude of the pulse.
        """
        if isinstance(amplitude, units.Value):
            return amplitude[units.Unit(amplitude)]
        else:
            return float(amplitude)
        
    def _harmonic(self, frequency, phase):
        """
        Process the pulse frequency and phase.
        
        Inputs:
            frequency: frequency of the harmonic pulse.
            phase: phase of the harmonic pulse.
        Outputs:
            frequency: frequency of the harmonic pulse.
            phase: phase of the harmonic pulse.
        """
        if isinstance(frequency, units.Value):
            frequency = frequency['GHz']
        else:
            frequency = float(frequency)
        
        if isinstance(phase, units.Value):
            phase = phase['rad']
        else:
            phase = float(phase)
            
        return frequency, phase
        
    def _check_pulse(self):
        """
        Check whether the pulse amplitudes are in -1.0 to 1.0 range.
        
        Input:
            None.
        Output:
            None.
        """
        if any(abs(self.pulse) > 1):
            raise ValueError('The pulse amplitude should not exceed 1.')
    
    def after(self, time=0):
        """
        Time point after the pulse.
        
        Input:
            time: time delay after this pulse in ns.

        Output:
            time: absolute time.
        """
        return self.end + 1 + self._ns(time)
        
    def before(self, time=0):
        """
        Time point before the pulse.
        
        Input:
            time: time delay before this pulse in ns.
        Output:
            time: absolute time.
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
        amplitude = self._amplitude(amplitude)
        
        self.pulse = np.full(self.duration, amplitude)
        self._check_pulse()


class Sine(_WavePulse):
    """
    Sine pulse.

    Inputs:
        amplitude: amplitude of the sine pulse (default: 0).
        frequency: frequency of the sine pulse (default: 0 Hz).
        phase: phase of the sine pulse (default: 0 rad).
        offset: constant dc offset of the sine pulse (default: 0).
        start: starting time of the sine pulse.
        duration: length of the sine pulse.
        end: ending time of the sine pulse.
        phase_ref: point in time that should have the specified
            phase (default: start pulse time).
    """
    def __init__(self, amplitude=0, frequency=0, phase=0, offset=0,
                 start=None, duration=None, end=None, phase_ref=None):
        self._init_times(start, duration, end)
        amplitude = self._amplitude(amplitude)
        frequency, phase = self._harmonic(frequency, phase)
        offset = self._amplitude(offset)

        if phase_ref is None:
            t0 = 0
        else:
            t0 = self.start - self._ns(phase_ref)

        t = np.linspace(t0, t0 + self.duration - 1, self.duration)
        self.pulse = (offset + amplitude *
                np.sin(2 * np.pi * frequency * t + phase))
        self._check_pulse()


class Cosine(_WavePulse):
    """
    Cosine pulse.

    Inputs:
        amplitude: amplitude of the cosine pulse (default: 0).
        frequency: frequency of the cosine pulse (default: 0 Hz).
        phase: phase of the cosine pulse (default: 0 rad).
        offset: constant dc offset of the cosine pulse (default: 0).
        start: starting time of the cosine pulse.
        duration: length of the cosine pulse.
        end: ending time of the cosine pulse.
        phase_ref: point in time that should have the specified
            phase (default: start pulse time).
    """
    def __init__(self, amplitude=0, frequency=0, phase=0, offset=0,
                 start=None, duration=None, end=None, phase_ref=None):
        self._init_times(start, duration, end)
        amplitude = self._amplitude(amplitude)
        frequency, phase = self._harmonic(frequency, phase)
        offset = self._amplitude(offset)

        if phase_ref is None:
            t0 = 0
        else:
            t0 = self.start - self._ns(phase_ref)

        t = np.linspace(t0, t0 + self.duration - 1, self.duration)
        self.pulse = (offset + amplitude *
                np.cos(2 * np.pi * frequency * t + phase))
        self._check_pulse()


class Gaussian(_WavePulse):
    """
    Gaussian window pulse. The pulse is truncated at about 1 per 2^14
    level since the DACs have 14-bit resolution.

    Inputs:
        amplitude: amplitude of the gaussian pulse.
        start: starting time of the gaussian pulse.
        duration: length of the gaussian pulse.
        end: ending time of the gaussian pulse.
    """
    def __init__(self, amplitude=0, start=None, duration=None, end=None):
        self._init_times(start, duration, end)
        amplitude = self._amplitude(amplitude)

        sigma = (float(self.duration) - 1) / np.sqrt(112 * np.log(2))
        self.pulse = amplitude * ss.gaussian(self.duration, sigma)
        self._check_pulse()


class FromArray(_WavePulse):
    """
    Generate a pulse from a numpy array. The start or end times can be
    arbitrary, and the duration is derived automatically from the length
    of the array
    
    Inputs:
        pulse_data: numpy array containing the pulse data in 1 ns
            chunks.
        start: starting time of the pulse.
        end: ending time of the pulse.
    """
    def __init__(self, pulse_data=[], start=None, end=None):
        duration = len(pulse_data)
        self._init_times(start, duration, end)
        
        if isinstance(pulse_data, list):
            pulse_data = np.array(pulse_data)
        
        self.pulse = pulse_data
        self._check_pulse()        


class Waveform():
    """
    Create a waveform from pulses.
    
    The start of one pulse is expected to be one unit
    (i.e. one nanosecond) after the end of the previous pulse
    (i.e. pulse2.end - pulse1.start >= 1). Therefore, to make pulse B
    start immediately after another pulse A initialize B.start to
    (A.end + 1), or simply assign A.after() to B.start.
    
    Inputs:
        label: waveform label string.
        args: arbitrarily long set of _WavePulses to create the waveform
            from. To create a _WavePulse use one of the "public"
            classes such as DC, Sine, Cosine, etc.
    """
    def __init__(self, label='None', *args):
        if not isinstance(label, str):
            raise ValueError('Invalid waveform label.')
        self.label = label
        
        args = list(_flatten(args))
        pulses = [arg for arg in args if isinstance(arg, _WavePulse)]

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

def ECLDuringPulses(*args, **kwargs):
    """
    Return _WavePulse to make ECL outputs go high during a set of 
    specified _WavePulses
    
    Inputs: 
        args: set (or list) of _WavePulses during which an ECL pulse 
            should be generated.
        pad_length: time before and after the pulses (default: 8 ns).
    Output:
        ECL: list of ECL _WavePulses.
    """
    if 'pad_length' in kwargs:
        if isinstance(kwargs['pad_length'], units.Value):
            pad_length = kwargs['pad_length']['ns']
        else:
            pad_length = kwargs['pad_length']
        try:
            pad_length = int(np.round(pad_length))
        except:
            raise Exception("Invalid ECL pad length value.")
    else:
        pad_length = 8
    args = list(_flatten(args))
    pulses = [arg for arg in args if isinstance(arg, _WavePulse)]
    ECL = []
    for pulse in pulses:
        ECL.append(DC(amplitude = 1,
                      start = pulse.before(pad_length),
                      end = pulse.after(pad_length)))
    return ECL

def Harmonic(amplitude=0, frequency=0, phase=0,
            cosine_offset=0, sine_offset=0,
            start=None, duration=None, end=None, phase_ref=None):
    """
    Return cosine and sine pulses.

    Inputs:
        amplitude: amplitude of the pulses  (default: 0).
        frequency: frequency of the pulses  (default: 0 Hz).
        phase: phase of the pulses  (default: 0 rad).
        cosine_offset: constant dc offset of the cosine pulse
            (default: 0).
        sine_offset: constant dc offset of the sine pulse
            (default: 0).
        start: starting time of the pulses.
        duration: length of the pulses.
        end: ending time of the pulses.
        phase_ref: point in time that should have the specified
            phase (default: start pulse time).
    
    Outputs:
        sine: Sine pulse object.
        cosine: Cosine pulse object.
    """
    return (Cosine(amplitude, frequency, phase,
            cosine_offset, start, duration, end, phase_ref),
              Sine(amplitude, frequency, phase,
              sine_offset, start, duration, end, phase_ref))

def wfs_dict(*args, **kwargs):
    """
    Return a waveform dictionary with the waveform labels as the keys.
    
    Align the waveforms using the waveform starting time. Ensure that
    the waveforms are of an equal length. The waveforms are zero-padded
    at the start and the end to ensure that they are not shorter than
    the minimum allowed length.
    
    Inputs:
        *args: arbitrarily long set of the Waveforms (instances of class
            Waveforms).
        *kwargs:
            min_length: minimum allowed length of the final waveform.
                Short waveforms are padded with zeros at the end 
                to increase their length (default: 20).
            start_zeros: number of zeros to add to the start of each
                waveform (default: 4).
            end_zeros: number of zeros to add to the end of each
                waveform (default: 4). Actual number of zeros added may
                be higher if the waveform length does not satisfy
                the min_length requirement.
    Outputs:
        waveforms: dictionary with the processed waveforms.
        offset: difference between the corresponding index values
            of the waveform numpy ndarrays and the time values that
            specify the start and end times for the waveforms:
            offset = ndarray_index - assigned_time_value, i.e.
            ndarray_index = assigned_time_value + offset.
    """
    defaults = {'min_length': 20, 'start_zeros': 4, 'end_zeros': 4}
    for key in kwargs:
        if isinstance(kwargs[key], units.Value):
            kwargs[key] = kwargs[key]['ns']
        try:
            kwargs[key] = int(np.round(kwargs[key]))
        except:
            raise Exception("Invalid parameter '%s' value." %key)
    defaults.update(kwargs)
    min_len = defaults['min_length']
    start, end = defaults['start_zeros'], defaults['end_zeros']

    wfs = [arg for arg in args if isinstance(arg, Waveform)]

    # Align the waveforms.
    if wfs:
        start_offset = min([wf.start for wf in wfs])
        for wf in wfs:
            wf.pulses = np.hstack([np.zeros(wf.start - start_offset),
                                   wf.pulses])
    else:
        start_offset = 0
   
    # Create an empty waveform 'None'.
    wfs.append(Waveform('None', DC(start=start_offset, duration=1)))
     
    # Ensure that the waveforms are long enough and of an equal length.
    max_len = max([wf.pulses.size for wf in wfs]) + start + end
    total_len = max(min_len, max_len)
    for wf in wfs:
        fin = max(total_len - start - wf.pulses.size, end)
        wf.pulses = np.hstack([np.zeros(start), wf.pulses, np.zeros(fin)])

    return {wf.label: wf.pulses for wf in wfs}, start - start_offset

def check_wfs(waveforms):
    """
    Check that all waveforms have the same length.
    
    Input:
        waveforms: dictionary with the processed waveforms.
    Output:
        None.
    """
    lengths = [waveforms[wf].size for wf in waveforms]
    if lengths.count(lengths[0]) != len(lengths):
        raise Exception('The waveform have different lengths.')
  
def _close_figure(self, signal=None):
        """
        Close the waveform figure.
        
        Input:
            None.
        Output:
            None.
        """
        plt.close(2)

def plot_wfs(waveforms, wf_labels, wf_colors=['r', 'g', 'm', 'b', 'k', 'c']):
    """
    Plot waveforms.
    
    Input:
        waveforms: dictionary with the processed waveforms.
        wf_labels: waveform labels to plot.
        wf_colors: colors for waveform colorcoding.
    Output:
        None.
    """
    if not isinstance(wf_colors, list):
        wf_colors = list(wf_colors)
    
    if not isinstance(wf_labels, list):
        wf_labels = list(wf_labels)

    time = waveforms[wf_labels[0]].size
    time = np.linspace(0, time - 1, time)
    plt.figure(2)
    plt.ioff()
    plt.clf()
    for idx, wf in enumerate(wf_labels):
        plt.plot(time, waveforms[wf], wf_colors[idx % 6],
                label=wf_labels[idx])
    plt.xlim(time[0], time[-1])
    plt.legend()
    plt.xlabel('Time [ns]')
    plt.ylabel('Waveforms')
    plt.draw()
    plt.pause(0.05)


if __name__ == "__main__":
    """
    Tests and examples. Add your test/example!
    """
    # Explicitly close the waveform figure when the terminal is closed.
    SetConsoleCtrlHandler(_close_figure, True)
    
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
    waveformB = Waveform('B', pulseB1, pulseB2)
    
    # Specifying the start, duration and end times at the same time will
    # work only if these parameters are consistent, i.e. if the equation
    # self.duration = self.end - self.start + 1 is satisfied.
    pulseA2 = DC(start=pulseB2.start, duration=10, end=pulseB2.end)
    try:
        # Inconsistent specifications.
        pulseA2 = DC(start=pulseB2.after(-1), duration=12, end=pulseB2.end)
    except ValueError:
        print('The inconsistent time error has been correctly caught.')
    
    try:
        # Amplitude should not exceed 1.
        pulseA2 = Sine(amplitude=1, frequency=.25, offset=.1,
                       start=pulseB2.after(-1), duration=12)
    except ValueError:
        print('The amplitude error has been correctly caught.')
    
    # Sine pulse with amplitude of 1 and frequency of 0.1 GHz
    # starting 2 ns after pulseB1 and ending at the same time as 
    # pulseB2.
    pulseA2 = Sine(amplitude=1, phase=np.pi/2, frequency=0.1,
                   start=pulseB1.after(2), end=pulseB2.end)
    
    # Combine the two pulses into one waveform. The waveform class
    # automatically puts the wave pulses in the correct order.
    waveformA = Waveform('A', pulseA1, pulseA2)
    
    # Create a waveform dictionary with the waveform labels as the keys.
    # The waveforms will be aligned based on their start times. They
    # will be zero-padded to ensure equal length that is longer than
    # a minimum length, which is 20 in this example.
    wfs, time_offset = wfs_dict(waveformA, waveformB, min_length=20)
    print(wfs)
    check_wfs(wfs)
    print('Time offset = %d ns.' %time_offset)

    # Gaussian pulse with amplitude of 1 starting at t = 0 ns and
    # ending at t = 14 ns (duration is equal to 15 ns).
    pulseC = Gaussian(amplitude=1, start=0, duration=15, end=14)
    waveformC = Waveform('C', pulseC)
    
    wfs, time_offset = wfs_dict(waveformA, waveformB, waveformC,
                                min_length=100)
    print(wfs)
    check_wfs(wfs)
    print('Time offset = %d ns.' %time_offset)

    # Create an in-phase and quadrature components of a harmonic pulse.
    I, Q = Harmonic(amplitude=0.25, frequency=0.05, start=0,
                    duration=150)
    wfs, time_offset = wfs_dict(Waveform('I', I), Waveform('Q', Q))
    print(wfs)
    check_wfs(wfs)
    print('Time offset = %d ns.' %time_offset)
    # Plot the waveforms for inspection.
    plot_wfs(wfs, ['I', 'Q'], ['r', 'b'])
    
    # Some animation.
    for x in range(100):
        # Create an in-phase and quadrature components of a harmonic
        # pulse.
        I, Q = Harmonic(amplitude=0.25, frequency=0.03, phase= x / 20,
                        start=0, duration=150)
        wfs, time_offset = wfs_dict(Waveform('I', I), Waveform('Q', Q))
        # Plot the waveforms for inspection.
        plot_wfs(wfs, ['I', 'Q'], ['r', 'b'])