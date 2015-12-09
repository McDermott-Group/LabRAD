#fpgaTest utility, to check functionality of FPGA ADC and DAC boards
#GR, 12/7/15

#CURRENTLY FOR USE ONLY ON MCDERMOTT5125-2

import os
import sys

SERVERS_PATH = r'C:\Users\McDermott Lab\Desktop\servers'
PAD_LENGTH = 10

sys.path.insert(0, SERVERS_PATH)

import warnings

import numpy as np

import matplotlib
try:
    matplotlib.use('GTKApp')
except:
    print 'Could not use GTKApp rendering'
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
import scipy.io as sio

import labrad
import labrad.units as units

import imp

import fpgalib.fpga as fpgaModule
import fpgalib.adc as adcModule
import fpgalib.dac as dacModule

print 'All imports completed successfully'

class BoardRunner(object):

    def __init__(self, fpga, name):
        
        self.fpga = fpga
        self.name = name
        self.sram = []
        self.mem  = []
        self.sram_start = 0
        self.ready = False
        
    def send(self):
        pass
    

class ADCRunner(BoardRunner):
    
    def _mixerScale(self, v):
        #mixer values are defined only between -128 and +127
        if any(np.abs(v) > 1.):
            raise Exception('mixer table must be in (-1,1)')
        return (np.fix(127.5*v - 0.5)).astype('int')
    
    def _emptyMixerTable(self):
        """
        Return mixer table set to 0
        """
        return np.zeros((512,2))
        
    def _hannWindow(self, t):
        return 0.5*(1-np.cos(2*np.pi*t/(t[-1]-1)))
    
    def _squareWindow(self, t):
        return np.ones(len(t))*1.
    
    def _flatTopWindow(self, t):
        x = 2*np.pi*t/(t[-1]-1)
        FT = (1 - 1.93 * np.cos(x) + 1.29 * np.cos(2 * x) - 
                0.388 * np.cos(3 * x) + 0.028 * np.cos(4 * x))
        return FT/np.max(FT)
        
        
    def _singleMixerTable(self, freq, plot=False):
        """
        Return mixer table with single frequency demodulation
        """
        t = np.arange(0,1024,2)
        if hasattr(self, 'window_function'):
            filt = self.window_function(t)
        else:    
            filt = self._squareWindow(t)
        I = self._mixerScale(np.cos(2*np.pi*t*freq)*filt) 
        Q = self._mixerScale(np.sin(2*np.pi*t*freq)*filt) 
        
        if plot:
            plt.figure(2)
            plt.plot(t, I, 'r-')
            plt.plot(t, Q, 'b-')
            plt.draw()

        return np.vstack([I,Q]).T
        
    def _onesMixerTable(self, length):
        t_max = 1024
        if length > t_max:
            length = t_max
        on = np.floor(length/2)
        off = 512-on
        return np.vstack([127*np.ones((on,2)), np.zeros((off,2))])
                
    
    def _getBuild(self):
        p = self.fpga.packet()
        p.select_device(self.name)
        p.build_number()
        resp = p.send()
        
        self.build = int(resp.build_number)
        
    def _send_build1(self):
        raise Exception('No Build 1 data yet...')
    
    def _send_build7(self):
        p = self.fpga.packet()
        p.select_device(self.name)
        p.adc_run_mode(self.mode)
        p.start_delay(int(self.start_delay))
        p.adc_trigger_table(self.trigger_table)
        for chan in range(12):
            p.adc_mixer_table(chan, self.mixer_tables[chan])
        p.send(wait=False)

    def send(self):
        if not self.ready:
            raise Exception('ADC SRAM not set!')
        
        self._getBuild()
        if self.build not in [1, 7]:
            raise Exception('Unrecognized ADC build')
            
        if self.build == 1:
            self._send_build1()
        elif self.build == 7:
            self._send_build7()
            
    def set_window_function(self, function):
        if function == 'hann':
            self.window_function = self._hannWindow
        elif function == 'square':
            self.window_function = self._squareWindow
        elif function == 'flattop':
            self.window_function = self._flatTopWindow
        else:
            raise Exception('Unknown window function %s'%function)
        
        
        
    def set_adc_average(self, startDelay): 
        
        self.start_delay = startDelay['ns']
        self.demod_frequency = 0
        self.mode = 'average'
            
        #self.trigger_table = [(1, 1, 4000, 12), (1, 100, 4000, 12)]
        self.trigger_table = [(1,100,256,12)]
        
        empty_table = self._emptyMixerTable()
        
        self.mixer_tables = [empty_table for _ in xrange(12)]
        
        self.ready = True
        
    def set_adc_demod(self, startDelay, demodFrequency):
        
        self.start_delay = startDelay['ns']
        self.demod_frequency = demodFrequency['GHz']
        self.mode = 'demodulate'
        
        #trig table is
        #(count, delay, length, rchan)
        self.trigger_table = [(3,50,256,1)]
        
        empty_table = self._emptyMixerTable()
        self.mixer_tables = [self._singleMixerTable(self.demod_frequency)]
        for _ in xrange(11):
            self.mixer_tables.append(empty_table)
        
        # for i in xrange(12):
            # print self.mixer_tables[i].shape
        
        self.ready = True
        
    def set_test_demod(self, startDelay, demodFreq, rdelay):
        self.start_delay = startDelay['ns']
        self.mode = 'demodulate'
        
        self.trigger_table = [(1,10,rdelay,12),]
        empty_table = self._emptyMixerTable()
        self.mixer_tables = [self._singleMixerTable(demodFreq['GHz'])]
        for _ in xrange(11):
            self.mixer_tables.append(empty_table)
        
        self.ready = True
        
        
        
    
        
class DACRunner(BoardRunner):
    
    def send(self):
        """send the data to the board"""
        if not self.ready:
            raise Exception('DAC SRAM and Memory not ready!!!')
        p = self.fpga.packet()
        p.select_device(self.name)
        p.memory(self.mem)
        p.sram(self.sram)
        p.send(wait=False)
                
    def _simpleMemory(self):
        """Build a simple memory sequence that runs SRAM"""
        
        sram_length = len(self.sram)
        self.mem = dacModule.MemorySequence()
        self.mem.noOp()
        self.mem.delayCycles(250 * 100) # delay 100 us
        self.mem.sramStartAddress(0)
        self.mem.sramEndAddress(sram_length-1)
        self.mem.runSram()
        self.mem.delayCycles(4000) #delay 16us
        self.mem.startTimer()
        self.mem.stopTimer()
        self.mem.branchToStart()
        
        # for val in self.mem:
            # print hex(val)
        
    def _waves2sram(self, waveA, waveB, trigger=True):
        """Construct SRAM sequence for a list of waveforms."""
        if not len(waveA) == len(waveB):
            raise Exception('DAC A and DAC B waveforms must be of an equal length.')
        if any(np.hstack((waveA, waveB)) > 1.0):
            raise Exception('The GHz DAC wave amplitude cannot exceed 1.0 [DAC Units].')
        dataA = [long(np.floor(0x1FFF * y)) for y in waveA]   # Multiply wave by full scale of DAC. DAC is 14 bit 2's compliment,
        dataB = [long(np.floor(0x1FFF * y)) for y in waveB]   # so full scale is 13 bits, i.e. 1 1111 1111 1111 = 1FFF.
        dataA = [y & 0x3FFF for y in dataA]                   # Chop off everything except lowest 14 bits.
        dataB = [y & 0x3FFF for y in dataB]
        dataB = [y << 14 for y in dataB]                      # Shift DAC B data by 14 bits.
        sram=[dataA[i] | dataB[i] for i in range(len(dataA))] # Combine DAC A and DAC B.
        
        #add a trigger pulse on ECL0
        if trigger:
            sram[4] |= (1 << 28)
            sram[5] |= (1 << 28)
            sram[6] |= (1 << 28)
            sram[7] |= (1 << 28)
            sram[8] |= (1 << 28)
            sram[9] |= (1 << 28)
            sram[10] |= (1 << 28)
        
        return sram
        
    def _sinewave(self, length, amplitude, frequency, phase, offset=0):
        
        t = np.linspace(0, int(length) - 1, int(length))
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
            
    def _pad(self):
        return np.zeros(PAD_LENGTH)
            
    def sine_output(self, start, length, amplitude, frequency, phase):
        
        zro = np.zeros(start['ns'])
        
        A = self._sinewave(length['ns'], amplitude, 
                frequency['GHz'], 0)
        B = self._sinewave(length['ns'], amplitude, 
                frequency['GHz'], phase['rad'])
                
        waveA = np.hstack([self._pad(), zro, A, self._pad()])
        waveB = np.hstack([self._pad(), zro, B, self._pad()])        
        
        self.sram = self._waves2sram(waveA, waveB)
        self._simpleMemory()
        self.ready = True

        
    
    
class boardTester(object):

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        # Disconnect from LabRAD.
        if hasattr(self, 'cxn'):
            self.cxn.disconnect()
        plt.close('all')
            
    def __init__(self, cxn):
    
        self.fpga = cxn.ghz_fpgas()
        
        adc_names = []
        dac_names = []
        boards = self.fpga.list_devices()
        
        self.reps = 1
        
        for board in boards:
            if 'adc' in board[1].lower():
                adc_names.append(board[1])
            if 'dac' in board[1].lower():
                dac_names.append(board[1])
                
        if len(adc_names) is not 1:
            raise Exception('Only 1 ADC allowed!')
        if len(dac_names) is not 1:
            raise Exception('Only 1 DAC allowed!')
            
        for adc in adc_names:
            self.adc = ADCRunner(self.fpga, adc)
        for dac in dac_names:
            self.dac = DACRunner(self.fpga, dac)
            
    def _run(self):
        self.dac.send()
        self.adc.send()
        daisyChainList = [self.dac.name] + [self.adc.name]
        self.fpga.daisy_chain(daisyChainList)
        if self.adc.mode == 'average':
            self.fpga.timing_order([self.adc.name])
        elif self.adc.mode == 'demodulate':
            self.fpga.timing_order([self.adc.name + '::0'])
        else:
            raise Exception('Unknown ADC mode.')
        
        data = self.fpga.run_sequence(self.reps, True)
        return data
        
    def average(self, reps=1, plot=True):
        
        print 'Running ADC in average mode.'
        
        self.reps = reps
        
        self.dac.sine_output(1 * units.us, 2 * units.us, 0.5,
                                3 * units.MHz, np.pi/2 * units.rad)
        
        self.adc.set_adc_average(0*units.ns)
        
        data = self._run()
        
        Is, Qs = data[0]
        
        if plot:
            plt.figure()
            t = np.linspace(0, 2 * (np.size(Is) - 1), np.size(Is))
            plt.plot(t, Is, '-')
            plt.plot(t, Qs,'r-')
            plt.xlabel('Time [ns]')
            plt.ylabel('ADC Amplitude [ADC bits]')
            plt.legend(['Is', 'Qs'])
            plt.show()
            
    def demod(self, reps=1, plot=True):
        print 'Running ADC demod test'
        
        self.reps = reps
        self.dac.sine_output(10 * units. ns, 3 * units.us, 0.5,
                                20 * units.MHz, np.pi/2 * units.rad)
                                
        self.adc.set_adc_demod(0*units.ns, -20 * units.MHz)
        
        data = self._run()
        
        print 'data shape: '
        print data.shape
        
        Is = [d[0][0] for d in data[0]]
        Qs = [d[0][1] for d in data[0]]       
        
        Is2 = [d[1][0] for d in data[0]]
        Qs2 = [d[1][1] for d in data[0]] 
        
        Is3 = [d[2][0] for d in data[0]]
        Qs3 = [d[2][1] for d in data[0]] 
        
        # Is,Qs = data[0]
        
        if plot:
            plt.figure()
            plt.plot(Is, Qs, 'r.')
            plt.plot(Is2, Qs2, 'b.')
            plt.plot(Is3, Qs3, 'g.')
            plt.xlabel('Is')
            plt.ylabel('Qs')
            plt.show()
            
    def demod_sweep(self, demod_freqs, dac_freq, reps=1, plot=True):
        print 'Running ADC demod frequency sweep'
        self.reps = reps;
        self.dac.sine_output(10 * units.ns, 8 * units.us, 0.5, dac_freq, np.pi/2*units.rad)
        
        Iavg = np.zeros(demod_freqs.shape)
        Qavg = np.zeros(demod_freqs.shape)
        
        #self.adc.set_window_function('flattop')
        
        for idx,f in enumerate(demod_freqs):
            print 'Running at demodulation frequency %f MHz'%f['MHz']
            self.adc.set_adc_demod(1*units.us, f)
            data = self._run()
            Is = [d[0][0] for d in data[0]]
            Qs = [d[0][1] for d in data[0]] 
            Iavg[idx] = np.mean(Is)
            Qavg[idx] = np.mean(Qs)
        
        Amp = np.sqrt(Iavg**2 + Qavg**2) 
        print np.max(Amp)
        if plot:
            plt.figure(1)
            plt.semilogy(demod_freqs['MHz'], Amp, 'r-')
            plt.xlabel('Demod Frequency (MHz)')
            plt.ylabel('ADC Amplitude')
            
            plt.show()

            
    def test_demod(self, dac_freq, reps=200, plot=True):
        print 'Running ADC Demod test'
        self.reps=reps
        self.dac.sine_output(0*units.ns, 8*units.ns, 0.5, dac_freq, np.pi/2*units.rad)
        
        dt = np.arange(2,256,2).astype('int')
        I = np.zeros(dt.shape)
        Q = np.zeros(dt.shape)
        for idx, t in enumerate(dt):
            print t
            self.adc.set_test_demod(0*units.ns, -1.*dac_freq, t)
            data = self._run()
            Is = data[0,:,0,0]
            Qs = data[0,:,0,1]
            I[idx] = np.mean(Is)
            Q[idx] = np.mean(Qs)
        
        A = np.sqrt(I**2 + Q**2)
        
        if plot:
            plt.figure()
            plt.plot(dt*4., A, 'r-')
            plt.xlabel('rlen cycles')
            plt.ylabel('ADC I/Q')
            plt.show()
            
            

#running...        

cxn = labrad.connect()
with boardTester(cxn) as b:
    
    b.average(reps=60)
    #b.demod(reps=1000)
    #b.demod_sweep( np.linspace(-40, 40, 101) * units.MHz, 20*units.MHz, reps=200)
    #b.test_demod(20 * units.MHz)