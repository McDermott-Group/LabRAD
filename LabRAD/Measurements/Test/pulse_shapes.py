# Copyright (C) 2012 Guilhem Ribeill
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
"""
Defines various useful GHz FPGA DAC pulse shapes for experiments.
"""

import numpy as np


def DC(length, amplitude):
    '''
    Returns a DC square pulse of a given length and amplitude.
    '''
    return np.zeros(int(length)) + amplitude

def SinePulse(length, frequency, amplitude, phase, offset):
    '''
    Returns a sine pulse.
    '''
    t = np.linspace(0, int(length) - 1, int(length))
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

def CosinePulse(length, frequency, amplitude, phase, offset):
    '''
    Returns a cosine pulse.
    '''
    t = np.linspace(0, int(length) - 1, int(length))
    return amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset

def GaussPulse(length, FW, amplitude):
    '''
    Returns a "slowed" square pulse that consists of gaussian rise and
    fall times and a DC segment. Length is the length of the DC part of
    the pulse in ns, and FW is the width at 1/10 maximum of the gaussian
    rise and fall. Total pulse length is thus ~= length + 2 * FW.
    '''
    c = FW / (2. * np.sqrt(2. * np.log(10.)))
    t = np.linspace(0, int(length + 3. * FW) - 1, int(length + 3. * FW))
    pls = np.zeros(len(t))
    t1 = t < FW
    t2 = (t >= FW) & (t < FW + length)
    t3 = (t >= FW + length)
    
    pls[t1] = amplitude * np.exp(-np.power(t[t1] - FW, 2.) / 2. / c**2.)
    pls[t2] = amplitude
    pls[t3] = amplitude * np.exp(-np.power(t[t3] - FW - length, 2.) / 2. / c**2.)
    return pls