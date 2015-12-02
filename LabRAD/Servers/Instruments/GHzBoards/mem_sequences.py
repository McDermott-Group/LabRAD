# Copyright (C) 2015 Guilhem Ribeill, Ivan Pechenezhskiy
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

import numpy as np

import labrad.units as units

import mem_commands as Mem


def _us(time):
    """
    Convert time to microseconds. Return an integer without any
    units attached.
    
    Input:
        time: time.
    Output:
        time: time in microseconds without any units attached.
    """
    if isinstance(time, units.Value):
        time = time['us']
    return int(np.round(time))
    
def _V(voltage):
    """
    Convert voltage to volts. Return a float number without any
    units attached.
    
    Input:
        voltage: voltage.
    Output:
        voltage: voltage in volts without any units attached.
    """
    if isinstance(voltage, units.Value):
        voltage = voltage['V']
    return float(voltage)


class MemSequence():
    """
    Build a memory sequence from a list of atomic operations.
    """ 
    def __init__(self):
        self.mem = []
        self.mem = Mem.AppendMemNoOp(self.mem)
        
        # FastBiasDACMode contains the current mode of the FastBias
        # DACs. FastBiasFirmware contains the firmware versions of
        # the FastBias cards on the corresponding fiber optic channels.
        # Recall that each GHz FPGA board has two fiber optic channels.
        self.FastBiasDACMode = {1: 'NotSelected', 2: 'NotSelected'}
        self.FastBiasFirmware = {1: '2.1', 2: '2.1'}
    
    def delay(self, time=0):
        """
        Delay in microseconds.
        
        Input:
            time: delay time in microseconds.
        """
        self.mem = Mem.AppendMemDelay(self.mem, _us(time))
        
    def firmware(self, channel=1, version='2.1'):
        """
        FastBias firmware version.
        
        Inputs:
            channel: fiberoptic channel, should be either 1 or 2.
            version: FastBias firmware version, should be either
                '1.0' or '2.1'.
        """
        self.FastBiasFirmware[channel] = version
     
    def switch(self, channel=1, mode='NotSelected', version='2.1'):
        """
        FastBias mode switch.
        
        Inputs:
            channel: fiberoptic channel, should be either 1 or 2.
            mode: FastBias DAC mode, should be either 'Fast', 'Slow', or
                'Fine' (or, alternatively, 'DAC1 Fast', 'DAC1 Slow',
                'DAC0').
            version: FastBias firmware version, should be either
                '1.0' or '2.1'.
        """
        self.FastBiasDACMode[channel] = mode
        if self.FastBiasFirmware[channel] != '2.1':
            self.mem = Mem.AppendMemSwitchDAC(self.mem, mode, channel)
            
    def bias(self, channel=1, voltage=0, mode='NotSelected'):
        """
        Set FastBias output voltage.
        
        Inputs:
            channel: fiberoptic channel, should be either 1 or 2.
            voltage: output voltage in volts.
            mode: FastBias DAC mode, should be either 'Fast', 'Slow', or
                'Fine' (or, alternatively, 'DAC1 Fast', 'DAC1 Slow',
                'DAC0').
        """
        voltage = _V(voltage)
        if mode != 'NotSelected':
            self.FastBiasDACMode[channel] = mode
            self.mem = Mem.AppendMemSetVoltage(self.mem, voltage, 
                    mode, channel, self.FastBiasFirmware[channel])
        elif (self.FastBiasDACMode[channel].lower().replace(' ', '') in
                ['dac0', 'fine', 'dac1slow', 'slow', 'dac1fast', 'fast']):
            self.mem = Mem.AppendMemSetVoltage(self.mem, voltage,
                    self.FastBiasDACMode[channel], channel,
                    self.FastBiasFirmware[channel])
        elif self.FastBiasDACMode[channel] == 'NotSelected':
            # This option is to maintain the backward compatibility.
            self.FastBiasDACMode[channel] = 'Fast'
            if self.FastBiasFirmware[channel] == '1.0':
                self.mem = Mem.AppendMemSwitchDAC(self.mem, 'Fast', channel)
            Mem.AppendMemSetVoltage(self.mem, voltage,
                    self.FastBiasDACMode[channel], channel,
                    self.FastBiasFirmware[channel])
        else:
            raise Exception("Invalid FastBias mode setting: " +
                    str(self.FastBiasDACMode[channel]) + '.')
                    
    def bias_then_wait(self, channel=1, voltage=0, mode='NotSelected',
        time=0):
        """
        Set FastBias output voltage.
        
        Inputs:
            channel: fiberoptic channel, should be either 1 or 2.
            voltage: output voltage in volts.
            mode: FastBias DAC mode, should be either 'Fast', 'Slow', or
                'Fine' (or, alternatively, 'DAC1 Fast', 'DAC1 Slow',
                'DAC0').
            time: delay time in microseconds.
        """
        self.bias(channel, voltage, mode)
        self.mem = Mem.AppendMemDelay(self.mem, _us(time))

    def sram(self, sram_length, sram_start=0):
        """
        Append SRAM memory commands.
        
        Inputs:
            sram_length: SRAM length.
            sram_start: SRAM start time.
        """
        self.mem = Mem.AppendMemSRAMStartAddress(self.mem, sram_start)
        self.mem = Mem.AppendMemSRAMEndAddress(self.mem, sram_start + 
                sram_length - 1)
        self.mem = Mem.AppendMemCallSRAM(self.mem)
        self.mem = Mem.AppendMemDelay(self.mem, np.ceil(sram_length / 1000))
        
    def timer(self, time=0):
        """
        Append timer commands.
        
        Inputs:
            time: stop time in microseconds.
        """       
        self.mem = Mem.AppendMemStartTimer(self.mem)
        self.mem = Mem.AppendMemDelay(self.mem, _us(time))
        self.mem = Mem.AppendMemStopTimer(self.mem)
    
    def sequence(self):
        """
        Append the end command.
        """
        # Automatically append the end command when the memory command
        # list is requested.
        return Mem.AppendMemEnd(self.mem)

def simple_sequence(init_time, sram_length, sram_start=0):
    """
    Build a simple memory sequence that waits initTime, then starts
    SRAM.
    
    Inputs:
        init_time: initialization time in microseconds.
        sram_length: SRAM length.
        sram_start: SRAM start time.
    """ 
    memory = []
    memory = Mem.AppendMemNoOp(memory)
    memory = Mem.AppendMemDelay(memory, _us(init_time))
    memory = Mem.AppendMemSRAMStartAddress(memory, sram_start)
    memory = Mem.AppendMemSRAMEndAddress(memory, sram_start + sram_length - 1)
    memory = Mem.AppendMemCallSRAM(memory)
    memory = Mem.AppendMemDelay(memory, np.ceil(sram_length / 1000))
    memory = Mem.AppendMemStartTimer(memory)
    memory = Mem.AppendMemStopTimer(memory)
    memory = Mem.AppendMemDelay(memory, 10)
    memory = Mem.AppendMemEnd(memory)
    return memory

def waves2sram(waveA, waveB, ECLdata=None):
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

    if ECLdata is not None:
        if not len(waveA) == len(ECLdata):
            raise Exception('ECL list should be the same length as the waveform')
        sram = [sram[i] | ECLdata[i] for i in range(len(dataA))]
    
    return sram

def _truncate(a):
        return int(a>0)
    
def waves2ECL(ECL_dict, trigs=[]):
    """Convert lists defining ECL output to a single 4-bit word for
    the ECL serializer."""
    
    # Check if all lists are empty.
    if all(len(x) == 0 for x in ECL_dict.values()):
        return None
    #add trigger output if needed        
    #Check if there is an unknown keyvalue in ECL dictionary
    if len([k for k in ECL_dict.keys() if k not in ['ECL0', 'ECL1', 'ECL2', 'ECL3']]) > 0:
        raise Exception('Unknown key in ECL data dictionary')
    length = max([len(x) for x in ECL_dict.values()])
    # Check that they are all the same length.
    for ecl in ECL_dict.iterkeys():
        if len(ECL_dict[ecl]) == 0:
            ECL_dict[ecl] = np.zeros((length,))
    if len([x for x in ECL_dict.values() if len(x) != length]) > 0:
        raise Exception('ECL data definitions should be of an equal length.')
    
    #convert to 1 or 0
    vtrunc = np.vectorize(_truncate)
    D0 = vtrunc(ECL_dict['ECL0'])
    D1 = vtrunc(ECL_dict['ECL1'])
    D2 = vtrunc(ECL_dict['ECL2'])
    D3 = vtrunc(ECL_dict['ECL3'])
    ECLdata = [(D0[i] << 28) | (D1[i] << 29) | (D2[i] << 30) | (D3[i] << 31) for i in xrange(length)]
    #Add a trigger pulse if it is needed.  Explanation of the code below:
    #trig is of the form '[ECLn, ]' where n is the ECL output we want the
    #triggers to appear on. The ECL bits are sram[31..28], so we or in a 
    #bit shifted by 28+n to the ECL data pulse
    if len(trigs) > 0:        
        for t in trigs:
            if int(t[-1]) not in [0, 1, 2, 3]:
                raise Exception("Invalid trigger definition: " + 
                                str(trig)) #safety first!
            ECLdata[4] |= (1 << (28+int(t[-1])))                     
            ECLdata[5] |= (1 << (28+int(t[-1])))
            ECLdata[6] |= (1 << (28+int(t[-1])))
            ECLdata[7] |= (1 << (28+int(t[-1])))
            ECLdata[8] |= (1 << (28+int(t[-1])))

    return ECLdata
    
    
    
    
    
    