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


ADC_DEMOD_CHANNELS = 4
SRAM_LEN = 18432
BLOCK0_LEN = 16384
BLOCK1_LEN = 2048
FILTER_LEN = 4096
SRAM_DELAY_LEN = 1024
DACDelays = [6, 2] # DAC delays in 4 ns chunks.
ADCDelays = [35]   # ADC delays

def memFromList(memList):
    """
    For more complex memory sequences, automatically build a memory sequence from a list of atomic operations.
    The memList should be a list of dictionaries, each dictionary representing a memory operation. Start and End 
    operations are added automatically.
    
    Each dictionary should have the format: {'Type': type_of_command, 'Value1', val1, 'Value2', val2, ...}
    
    Recognized dictionaries:
        {'Type': 'Delay', 'Time': delay_time_in_us}
        {'Type': 'Switch', 'Channel': channel_number, 'Mode': mode}
        {'Type': 'Bias', 'Channel': channel_number, 'Voltage': voltage_in_volts, ['Mode': mode]}
        {'Type': 'BiasThenWait', 'Channel': channel_number, 'Voltage': voltage_in_volts, 'Time': time_to_wait_at_voltage, ['Mode': mode]}
        {'Type': 'SRAM', 'Start': SRAM_start_address, 'Length': SRAM_length, 'Delay': SRAM_delay}
        {'Type': 'Timer', 'Time': time_to_poll_for_responses}
        {'Type': 'Firmware', 'Channel': channel_number, 'Version': version}
    """ 
    mem = []
    mem = Mem.AppendMemNoOp(mem)

    # FastBiasDACMode contains the current mode of the FastBias DACs: {Channel: mode,...}. 
    # FastBiasFirmware contains the firmware versions of the FastBias DACs: {Channel: firmware_version,...}. 
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
                mem = Mem.AppendMemSwitchDAC(mem, memOp['Mode'], memOp['Channel'])
                
        elif memOp['Type'] in ['Bias', 'BiasThenWait']:
            if 'Mode' in memOp:
                mem = Mem.AppendMemSetVoltage(mem, memOp['Voltage'], memOp['Mode'], memOp['Channel'], FastBiasFirmware[memOp['Channel']])
            elif FastBiasDACMode[memOp['Channel']].lower() in ['dac0', 'fine', 'dac1slow', 'slow', 'dac1fast', 'fast']:
                mem = Mem.AppendMemSetVoltage(mem, memOp['Voltage'], FastBiasDACMode[memOp['Channel']], memOp['Channel'], FastBiasFirmware[memOp['Channel']])
            elif FastBiasDACMode[memOp['Channel']] == 'NotSelected':        # This option is to maintain the backward compatibility.
                FastBiasDACMode[memOp['Channel']] = 'Fast'
                if FastBiasFirmware[memOp['Channel']] == '1.0':
                    mem = Mem.AppendMemSwitchDAC(mem, 'Fast', memOp['Channel'])
                Mem.AppendMemSetVoltage(mem, memOp['Voltage'], FastBiasDACMode[memOp['Channel']], memOp['Channel'], FastBiasFirmware[memOp['Channel']])
            else:
                raise Exception("Invalid FastBias 'Mode' setting: " + FastBiasDACMode[memOp['Channel']] + '.')
            if memOp['Type'] == 'BiasThenWait':
                mem = Mem.AppendMemDelay(mem, memOp['Time'])
            
        elif memOp['Type'] == 'SRAM':
            mem = Mem.AppendMemSRAMStartAddress(mem, memOp['Start'])
            mem = Mem.AppendMemSRAMEndAddress(mem, memOp['Start'] + memOp['Length'] - 1)
            mem = Mem.AppendMemCallSRAM(mem)
            mem = Mem.AppendMemDelay(mem, memOp['Delay'])
        
        elif memOp['Type'] == 'Timer': 
            mem = Mem.AppendMemStartTimer(mem)
            mem = Mem.AppendMemDelay(mem, memOp['Time'])
            mem = Mem.AppendMemStopTimer(mem)
            
        else:
            raise Exception("Unrecognized operation '" + memOp['Type'] + "' is specified in the memory list.")

    mem = Mem.AppendMemEnd(mem)
    
    return mem

def memSimple(initTime, SRAMLength, SRAMStart, SRAMDelay):
    """
    Build a simple memory sequence that waits initTime, then starts SRAM.
    """
    memory = []
    memory = Mem.AppendMemNoOp(memory)
    memory = Mem.AppendMemDelay(memory, initTime)
    memory = Mem.AppendMemSRAMStartAddress(memory,SRAMStart)
    memory = Mem.AppendMemSRAMEndAddress(memory,SRAMStart+SRAMLength-1)
    memory = Mem.AppendMemCallSRAM(memory)
    memory = Mem.AppendMemDelay(memory,SRAMDelay)
    memory = Mem.AppendMemStartTimer(memory)
    memory = Mem.AppendMemStopTimer(memory)
    memory = Mem.AppendMemDelay(memory,10)
    memory = Mem.AppendMemEnd(memory)
    
    return memory

def memBias(initTime, SRAMLength, SRAMStart, SRAMDelay, BiasVoltage, BiasTime):
    """
    Build a memory sequence that sets the fast bias card to a fixed voltage (BiasVoltage) waits BiasTime, then calls SRAM.
    Voltage returns to 0 at end of sequence.
    """
    mem = []
    
    #start
    mem = Mem.AppendMemNoOp(mem)
    
    #switch to +/-2.5V DACs, set voltage to zero and wait initTime
    mem = Mem.AppendMemSwitchDAC(mem,'DAC1Fast',1)
    mem = Mem.AppendMemSwitchDAC(mem,'DAC1Fast',2)
    mem = Mem.AppendMemSetDAC1Voltage(mem, 0, 1)
    mem = Mem.AppendMemSetDAC1Voltage(mem, 0, 2)
    mem = Mem.AppendMemDelay(mem, initTime)
    
    #switch to desired voltage, then wait appropriate time
    mem = Mem.AppendMemSetDAC1Voltage(mem, BiasVoltage, 1)
    mem = Mem.AppendMemSetDAC1Voltage(mem, BiasVoltage, 2)
    mem = Mem.AppendMemDelay(mem, BiasTime)
    
    #call SRAM and wait for it to execute before returning to zero voltage state
    mem = Mem.AppendMemSRAMStartAddress(mem, SRAMStart)
    mem = Mem.AppendMemSRAMEndAddress(mem, SRAMStart+SRAMLength-1)
    mem = Mem.AppendMemCallSRAM(mem)
    mem = Mem.AppendMemDelay(mem,SRAMDelay)
    mem = Mem.AppendMemStartTimer(mem)
    mem = Mem.AppendMemStopTimer(mem)
    mem = Mem.AppendMemSetDAC1Voltage(mem, 0, 1) #removed these two lines to prevent resetting to zero
    mem = Mem.AppendMemSetDAC1Voltage(mem, 0, 2)
    mem = Mem.AppendMemDelay(mem,10)
    mem = Mem.AppendMemEnd(mem)
    
    return mem

def LoadDACs(fpga,memory,sram,DACs):
    for k,val in enumerate(DACs):
        fpga.select_device(DACs[k])
        fpga.memory(memory[k])
        fpga.start_delay(DACDelays[k])
        ###handle dual block calls here, in a different way than sank in the fpga_Server--should be compatable
        if len(sram[k])>SRAM_LEN:
            SRAM_1 = sram[k][-BLOCK1_LEN:] #shove last chunk of SRAM into BLOCK1, be sure this can contain what you need it to contain 
            SRAM_diff = len(sram[k])-SRAM_LEN #amount of SRAM that's extra
            x,y = divmod(SRAM_diff,SRAM_DELAY_LEN) #SRAM_diff = x*SRAM_DELAY_LEN+y
            if y == 0:
                NumDelayBlocks = x
            else:
                NumDelayBlocks = x+1 #overshoot 
            SRAM_0 = sram[k][:(BLOCK0_LEN + SRAM_diff-NumDelayBlocks*SRAM_DELAY_LEN)]
            if len(set(sram[k][(BLOCK0_LEN + SRAM_diff-NumDelayBlocks*SRAM_DELAY_LEN)-4:len(sram[k])-BLOCK1_LEN]))!=1:
                raise Exception('Dual block mode will not work for this pulse sequence')    #checks to see if delay block is constant
            fpga.sram_dual_block(SRAM_0,SRAM_1,NumDelayBlocks*SRAM_DELAY_LEN)
        else:
            fpga.sram(sram[k])
            
def LoadADCs(fpga, ADCs, ADCVars):
    """
    Load ADCs with correct variables.
    """
    #GR, 5/15/2015
    #Updated to allow for more than 1 ADC -- it is assumed that ADC and ADCVars indicies correspond.
    for idx, ADC in enumerate(ADCs):
        fpga.select_device(ADC)
        fpga.start_delay(int((ADCVars[idx]['ADCDelay'])/4)+ADCDelays[0])
        fpga.adc_run_mode(ADCVars[idx]['RunMode'])
        fpga.adc_filter_func(filterBytes(ADCVars[idx]), ADCVars[idx]['filterStretchLen'], ADCVars[idx]['filterStretchAt'])
        for k in range(ADC_DEMOD_CHANNELS):
            dPhi = int(ADCVars[idx]['DemodFreq']/7629)
            phi0 = int(ADCVars[idx]['DemodPhase']*(2**16))
            fpga.adc_demod_phase(k, dPhi, phi0)
            fpga.adc_trig_magnitude(k, ADCVars[idx]['DemodSinAmp'], ADCVars[idx]['DemodCosAmp'])
        
def filterBytes(ADCVars):
    ### ADC collects at a 2 ns acquisition rate, but the filter function has a 4 ns resolution
    ### This function sets the filter for the specific experiment
    filterFunc = ADCVars['FilterType']
    sigma = ADCVars['FilterWidth']
    window = np.zeros(int(ADCVars['FilterLength']/4))
    if filterFunc=='square':
        window = window+(128)
        filt = np.append(window,np.zeros(FILTER_LEN-len(window)))
        filt = np.array(filt,dtype='<u1')
    elif filterFunc=='gaussian':
        env = np.linspace(-0.5,0.5,len(window))
        env = np.floor(128*np.exp(-((env/(2*sigma))**2)))
        filt =  np.append(env,np.zeros(FILTER_LEN-len(env)))
        filt = np.array(filt,dtype='<u1')        
    elif filterFunc=='hann':
        env = np.linspace(0,len(window)-1,len(window))
        env = np.floor(128*np.sin(np.pi*env/(len(window)-1))**2)
        filt =  np.append(env,np.zeros(FILTER_LEN-len(env)))
        filt = np.array(filt,dtype='<u1')
    elif filterFunc=='exp':
        env = np.linspace(0,(len(window)-1)*4,len(window))
        env = np.floor(128*np.exp(-env/sigma))
        filt =  np.append(env,np.zeros(FILTER_LEN-len(env)))
        filt = np.array(filt,dtype='<u1')
    else:
        raise Exception('Filter function %s not recognized' %filterFunc)
    return filt.tostring()

def waves2sram(waveA, waveB, Trig=True):
    """Construct SRAM sequence for a list of waveforms."""
    if not len(waveA) == len(waveB):
        raise Exception('Lengths of DAC A and DAC B waveforms must be equal.')
    if any(np.hstack((waveA, waveB)) > 1.0):
        raise Exception('The GHz DAC wave amplitude cannot exceed 1.0 [DAC units].')
    dataA = [long(np.floor(0x1FFF * y)) for y in waveA]   # Multiply wave by full scale of DAC. DAC is 14 bit 2's compliment,
    dataB = [long(np.floor(0x1FFF * y)) for y in waveB]   # so full scale is 13 bits, i.e. 1 1111 1111 1111 = 1FFF
    dataA = [y & 0x3FFF for y in dataA]                   # Chop off everything except lowest 14 bits.
    dataB = [y & 0x3FFF for y in dataB]
    dataB = [y << 14 for y in dataB]                      # Shift DAC B data by 14 bits.
    sram=[dataA[i] | dataB[i] for i in range(len(dataA))] # Combine DAC A and DAC B.
    if Trig:
        sram[4] |= 0xF0000000                             # Add trigger pulse near beginning of sequence.
        sram[5] |= 0xF0000000
        sram[6] |= 0xF0000000
        sram[7] |= 0xF0000000
        sram[8] |= 0xF0000000                             # Add trigger pulse near beginning of sequence.
    return sram