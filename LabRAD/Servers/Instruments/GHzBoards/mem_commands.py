# Copyright (C) 2010 David Hover
#           (C) 2015 Ivan Pechenezhskiy
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


def AppendMemNoOp(mem):
    mem.append(0x000000)
    return mem

def AppendMemEnd(mem):
    mem.append(0xF00000)
    return mem

def AppendMemDelay(mem, time):
    # Time should be specified in microseconds.
    mem.append(0x300000 + 25 * (int)(time))
    return mem

def AppendMemStartTimer(mem):
    mem.append(0x400000)
    return mem

def AppendMemStopTimer(mem):
    mem.append(0x400001)
    return mem

def AppendMemSRAMStartAddress(mem, address):
    mem.append(0x800000 + address)
    return mem

def AppendMemSRAMEndAddress(mem, address):
    mem.append(0xA00000 + address)
    return mem

def AppendMemCallSRAM(mem):
    mem.append(0xC00000)
    return mem

def _AddFOChannel(cmd, channel):
    if channel == 1:
        return cmd + 0x100000
    elif channel == 2:
        return cmd + 0x200000
    else:
        raise("Invalid fiber optic output channel: " + str(channel) + 
                ". The supported channels are 0 and 1.")
    
def AppendMemSwitchDAC(mem, mode='Fast', channel=1):
    mode_lowercase = mode.lower().replace(' ', '')
    if mode_lowercase in ['dac0', 'fine']:
        a = 0x50000
    elif mode_lowercase in ['dac1slow', 'slow']:
        a = 0x50002
    elif mode_lowercase in ['dac1fast', 'fast']:
        a = 0x50001
    else:
        print("Warning: mode '" + str(mode) + "' is not recognized. " +
                "The memory command will be ignored.")
        return mem
    mem.append(_AddFOChannel(a, channel))
    AppendMemDelay(mem, 5)
    return mem

def _AppendMemSetVoltage_v1p0(mem, voltage=0, mode='Fast', channel=1):
    mode_lowercase = mode.lower().replace(' ', '')
    if mode_lowercase in ['dac0', 'fine']:
        if (voltage < 0) or (voltage > 2.5):
            voltage = np.clip(voltage, 0, 2.5)
            print("Warning: FastBias DAC0 voltage cannot be set to a " +
                    "value beyond 0 and 2.5 V range. Voltage is set " +
                    "to " + str(voltage) + " V.")
        a = 0x60000 + (int)(voltage / 2.5 * 0xFFFF)
    elif mode_lowercase in ['dac1slow', 'slow', 'dac1fast', 'fast']:
        if (voltage < -2.5) or (voltage > 2.5):
            voltage = np.clip(voltage, -2.5, 2.5)
            print("Warning: FastBias DAC1 voltage cannot be set to a " +
                    "value beyond -2.5 and 2.5 V range. Voltage is " + 
                    "set to " + str(voltage) + " V.")
        a = 0x70000 + (int)((voltage / 2.5 + 1) * 0x7FFF)
    else:
        print("Warning: mode '" + str(mode) + "' is not recognized. " +
                "The memory command will be ignored.")
        return mem
    mem.append(_AddFOChannel(a, channel))
    AppendMemDelay(mem, 6)
    return mem

def _AppendMemSetVoltage_v2p1(mem, voltage=0, mode='Fast', channel=1):
    mode_lowercase = mode.lower()
    if mode_lowercase in ['dac0', 'fine']:
        dac = 0
        slew = 0
    elif mode_lowercase in ['dac1slow', 'slow']:
        dac = 1
        slew = 1
    elif mode_lowercase in ['dac1fast', 'fast']:
        dac = 1
        slew = 0
    else:
        print("Warning: mode '" + str(mode) + "' is not recognized. " +
                "The memory command will be ignored.")
        return mem
    if dac:
        if (voltage < -2.5) or (voltage > 2.5):
            voltage = np.clip(voltage, -2.5, 2.5)
            print("Warning: FastBias DAC1 voltage cannot be set to a " +
                    "value beyond -2.5 and 2.5 V range. Voltage is " +
                    "set to " + str(voltage) + " V.")
        data = (int)((voltage / 2.5 + 1) * 0x7FFF)
    else:
        if (voltage < 0) or (voltage > 2.5):
            voltage = np.clip(voltage, 0, 2.5)
            print("Warning: FastBias DAC0 voltage cannot be set to a " +
                    "value beyond 0 and 2.5 V range. Voltage is " +
                    "set to " + str(voltage) + " V.")
        data = (int)(voltage / 2.5 * 0xFFFF)
    cmd = (dac << 19) + (data << 3) + (slew << 2)
    mem.append(_AddFOChannel(cmd, channel))
    mem.append(0x300068)
    return mem

def AppendMemSetVoltage(mem, voltage=0, mode='Fast', channel=1,
                        firmware='2.1'):
    if firmware == '2.1':
        return _AppendMemSetVoltage_v2p1(mem, voltage, mode, channel)
    elif firmware == '1.0':
        return _AppendMemSetVoltage_v1p0(mem, voltage, mode, channel)
    else:
        raise Exception('FastBias firmware version ' + str(firmware) + 
                ' is not supported.')