# Copyright (C) 2015 Guilhem Ribeill, Ivan Pechenezhskiy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import mem_commands as Mem

def mem_from_list(memList):
    """
    For more complex memory sequences, automatically build a memory
    sequence from a list of atomic operations. The memList should be
    a list of dictionaries, each dictionary representing a memory
    operation. Start and End operations are added automatically.
    
    Each dictionary should have the format: {'Type': type_of_command,
        'Value1', val1, 'Value2', val2, ...}
    
    Recognized dictionaries:
        {'Type': 'Delay', 'Time': delay_time_in_us}
        {'Type': 'Switch', 'Channel': channel_number, 'Mode': mode}
        {'Type': 'Bias', 'Channel': channel_number,
            'Voltage': voltage_in_volts, ['Mode': mode]}
        {'Type': 'BiasThenWait', 'Channel': channel_number,
            'Voltage': voltage_in_volts, 
            'Time': time_to_wait_at_voltage, ['Mode': mode]}
        {'Type': 'SRAM', 'Start': SRAM_start_address,
            'Length': SRAM_length, 'Delay': SRAM_delay}
        {'Type': 'Timer', 'Time': time_to_poll_for_responses}
        {'Type': 'Firmware', 'Channel': channel_number,
            'Version': version}
    """ 
    mem = []
    mem = Mem.AppendMemNoOp(mem)

    # FastBiasDACMode contains the current mode of the FastBias DACs:
    # {Channel: mode,...}. 
    # FastBiasFirmware contains the firmware versions of the FastBias
    # DACs: {Channel: firmware_version,...}. 
    FastBiasDACMode = {1: 'NotSelected', 2: 'NotSelected'}
    FastBiasFirmware = {1: '2.1', 2: '2.1'}
    for memOp in memList:
        if memOp['Type'] == 'Delay':
            mem = Mem.AppendMemDelay(mem, memOp['Time'])

        elif memOp['Type'] == 'Firmware':
            FastBiasFirmware[memOp['Channel']] = memOp['Version']
            
        elif memOp['Type'] == 'Switch':
            FastBiasDACMode[memOp['Channel']] = memOp['Mode']
            if FastBiasFirmware[memOp['Channel']] != '2.1':
                mem = Mem.AppendMemSwitchDAC(mem, memOp['Mode'],
                                                  memOp['Channel'])
                
        elif memOp['Type'] in ['Bias', 'BiasThenWait']:
            if 'Mode' in memOp:
                FastBiasDACMode[memOp['Channel']] = memOp['Mode']
                mem = Mem.AppendMemSetVoltage(mem, memOp['Voltage'], 
                        memOp['Mode'], memOp['Channel'],
                        FastBiasFirmware[memOp['Channel']])
            elif (FastBiasDACMode[memOp['Channel']].lower() in
                    ['dac0', 'fine', 'dac1slow', 'slow', 'dac1fast', 'fast']):
                mem = Mem.AppendMemSetVoltage(mem, memOp['Voltage'],
                        FastBiasDACMode[memOp['Channel']],
                        memOp['Channel'],
                        FastBiasFirmware[memOp['Channel']])
            elif FastBiasDACMode[memOp['Channel']] == 'NotSelected':
                # This option is to maintain the backward compatibility.
                FastBiasDACMode[memOp['Channel']] = 'Fast'
                if FastBiasFirmware[memOp['Channel']] == '1.0':
                    mem = Mem.AppendMemSwitchDAC(mem, 'Fast', memOp['Channel'])
                Mem.AppendMemSetVoltage(mem, memOp['Voltage'],
                        FastBiasDACMode[memOp['Channel']],
                        memOp['Channel'],
                        FastBiasFirmware[memOp['Channel']])
            else:
                raise Exception("Invalid FastBias 'Mode' setting: " +
                        FastBiasDACMode[memOp['Channel']] + '.')
            if memOp['Type'] == 'BiasThenWait':
                mem = Mem.AppendMemDelay(mem, memOp['Time'])
            
        elif memOp['Type'] == 'SRAM':
            mem = Mem.AppendMemSRAMStartAddress(mem, memOp['Start'])
            mem = Mem.AppendMemSRAMEndAddress(mem, memOp['Start'] + 
                                                   memOp['Length'] - 1)
            mem = Mem.AppendMemCallSRAM(mem)
            mem = Mem.AppendMemDelay(mem, memOp['Delay'])
        
        elif memOp['Type'] == 'Timer': 
            mem = Mem.AppendMemStartTimer(mem)
            mem = Mem.AppendMemDelay(mem, memOp['Time'])
            mem = Mem.AppendMemStopTimer(mem)
            
        else:
            raise Exception("Unrecognized operation '" + memOp['Type'] + 
                    "' is specified in the memory list.")
    
    mem = Mem.AppendMemEnd(mem)
    
    return mem

def mem_simple(initTime, SRAMLength, SRAMStart, SRAMDelay):
    """
    Build a simple memory sequence that waits initTime, then starts
    SRAM.
    """
    memory = []
    memory = Mem.AppendMemNoOp(memory)
    memory = Mem.AppendMemDelay(memory, initTime)
    memory = Mem.AppendMemSRAMStartAddress(memory, SRAMStart)
    memory = Mem.AppendMemSRAMEndAddress(memory, SRAMStart + SRAMLength - 1)
    memory = Mem.AppendMemCallSRAM(memory)
    memory = Mem.AppendMemDelay(memory, SRAMDelay)
    memory = Mem.AppendMemStartTimer(memory)
    memory = Mem.AppendMemStopTimer(memory)
    memory = Mem.AppendMemDelay(memory, 10)
    memory = Mem.AppendMemEnd(memory)
    
    return memory

def waves2sram(waveA, waveB, ECL=[], Trig=True):
    """Construct SRAM sequence for a list of waveforms.
        Optional input for controlling the ECL outputs."""
    if not len(waveA) == len(waveB):
        raise Exception('Lengths of DAC A and DAC B waveforms must be equal.')
    if any(np.hstack((waveA, waveB)) > 1.0):
        raise Exception('The GHz DAC wave amplitude cannot exceed 1.0 [DAC units].')
        
    #if we are using the ECL outputs, make sure they are the correct length
    #and use the correct format
    if not len(ECL) == 0:
        if not len(ECL) == len(waveA):
            raise Exception('Length of ECL data must be the same as length of waveform data.')
        
    else:
        ECL = np.zeros(waveA.shape)
    
    dataA = [long(np.floor(0x1FFF * y)) for y in waveA]   # Multiply wave by full scale of DAC. DAC is 14 bit 2's compliment,
    dataB = [long(np.floor(0x1FFF * y)) for y in waveB]   # so full scale is 13 bits, i.e. 1 1111 1111 1111 = 1FFF.
    dataA = [y & 0x3FFF for y in dataA]                   # Chop off everything except lowest 14 bits.
    dataB = [y & 0x3FFF for y in dataB]
    ECLdata = [y & 0xF for y in ECLdata]                  #for ECL data, keep only lowest 4 bits (1 bit per channel)
    dataB = [y << 14 for y in dataB]                      # Shift DAC B data by 14 bits.
    ECLdata = [y << 28 for y in ECLdata]                  # shift ECL serial data by 28 bits.
    sram=[dataA[i] | dataB[i] | ECLdata[i] for i in range(len(dataA))] # Combine DAC A and DAC B and ECL data
    if Trig:
        sram[4] |= 0xF0000000                             # Add trigger pulse near beginning of sequence.
        sram[5] |= 0xF0000000
        sram[6] |= 0xF0000000
        sram[7] |= 0xF0000000
        sram[8] |= 0xF0000000                             # Add trigger pulse near beginning of sequence.
    
    return sram
    
def serial2ECL(ECL0=[], ECL1=[], ECL2=[], ECL3=[])
    """Convert lists defining ECL output to a single 4-bit word for the ECL serializer"""
    ECLlist = [ECL0, ECL1, ECL2, ECL3]
    
    #check if all lists are empty
    if all(len(x) == 0 for x in ECLlist):
        return []
    length = max([len(x) for x in ECLlist)
    #check that they are all the same length
    if len([x for x in ECLlist if len(x) != length]) > 0:
        raise Exception('Length of all ECL data definitons should be equal.')
    for idx, ecl in enumerate(ECLlist):
        if len(ecl) == 0:
            ECLlist[idx] = np.zeros((length,))
    ECLdata = np.zeros((length,))
    return []
        
    