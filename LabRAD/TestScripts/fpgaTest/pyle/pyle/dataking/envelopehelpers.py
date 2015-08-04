import numpy as np

import pyle.envelopes as env
from pyle.dataking import utilMultilevels as ml

import util

# sequence helpers
# these are simple functions for building standard control envelopes by
# pulling out appropriate configuration parameters for a particular qubit.

def power2amp(power):
    """
    Convert readout rms uwave power in to DAC amplitude.This function assumes a nonzero sb_freq.
    
    This function contains several hardcoded values and should be reworked.
    """
    assert power.isCompatible('dBm'), 'Power must be put in dBm.'
    power = power['dBm']
    #v = 10**(((power-10)/10.0+2)/2) #old shit
    Z=50 #impedance of system
    mixingAndCablingLossInDb=30 #loss due to mixing, coupling, filters etc... on the DAC board 30 dB typically: 10 from the IQ mixer, 10 from the directional coupler and 10 from attenuators
    rmsv = np.sqrt(Z * 10 **( (power+mixingAndCablingLossInDb) /10) * 1e-3   ) #power is rms power in dBm, i.e. 0 dBm = 1 mW
    v = rmsv*np.sqrt(2) #amplitude of voltage, non-rms
    dacamp = v/0.4 # dac_amp = 1 corresponds to 400mV
    if dacamp>1.0:
        print 'dacamp too big: ', dacamp
    elif dacamp<(0.03125):
        print 'dacamp too small (using less then 8 bits): ', dacamp
    #else:
        #print 'dacamp: ', dacamp
    return  dacamp


def mix(q, seq, freq=None, state=None):
    """Apply microwave mixing to a sequence.    
    This mixes to a particular frequency from the carrier frequency.
    Also, adjusts the microwave phase according to the phase calibration.
    
    PARAMETERS
    q: Qubit dictionary.
    seq - eh functions: Pulses to mix with microwaves.
    freq - string: Registry key indicating desired frequency of post-mix
           pulse (e.g., 'f10','f21').
    state - scalar: Which qubit frequency is desired for post-mix pulse
            (e.g., 1 gives f10, 2 gives f21).
    """
    if freq is not None and state is not None:
        raise Exception('state and freq are not orthogonal parameters for mixing')
    if isinstance(freq, str):
        freq = q[freq]
    if freq is None:
        if state is None:
            state=1
        freq = ml.getMultiLevels(q,'frequency',state)
    return env.mix(seq, freq - q['fc']) * np.exp(1j*q['uwavePhase'])


# xy rotations with half-derivative term on other quadrature
def piPulseHD(q, t0, phase=0, alpha=0.5, state=1, length='piFWHM'):
    """Pi pulse using a gaussian envelope with half-derivative Y quadrature."""
    return rotPulseHD(q, t0, angle=np.pi, phase=phase, state=state, length=length)

def piHalfPulseHD(q, t0, phase=0, alpha=0.5, state=1, length='piFWHM'):
    """Pi/2 pulse using a gaussian envelope with half-derivative Y quadrature."""
    return rotPulseHD(q, t0, angle=np.pi/2, phase=phase, state=state, length=length)

def rotPulseHD(q, t0, angle=np.pi, phase=0, alpha=0.5, state=1, length='piFWHM'):
    """Rotation pulse using a gaussian envelope with half-derivative Y quadrature.
    
    This also allows for an arbitrary pulse length. The new length must be defined as a key in the registry.
    """
    # Eliminate DRAG for higher order pulses
    if state>1: alpha = 0
    #Get the pi amplitude. getMultiLevels() ensures that the correct key is read regardless of which state is desired.
    #Note in particular that old code, which does not explicitly set state, and therefore gets the default value of 1,
    #will get 'piAmp', as desired. 
    piamp = ml.getMultiLevels(q,'piAmp',state)
    r = angle / np.pi
    delta = 2*np.pi * (q['f21'] - q['f10'])['GHz']
    x = env.gaussian(t0, w=q[length], amp=piamp*r, phase=phase)
    y = -alpha * env.deriv(x) / delta
    return x + 1j*y

def rabiPulseHD(q, t0, len, w=None, amp=None, overshoot=0.0, overshoot_w=1.0, alpha=0.5, state=1):
    """Rabi pulse using a flattop envelope with half-derivative Y quadrature."""
    # Eliminate DRAG for higher order pulses
    if state>1: alpha = 0
    #Get the pi amplitude. getMultiLevels() ensures that the correct key is read regardless of which state is desired.
    #Note in particular that old code, which does not explicitly set state, and therefore gets the default value of 1,
    #will get 'piAmp', as desired.
    if amp is None:
        amp = ml.getMultiLevels(q,'piAmp',state)
    if w is None:
        w=q['piFWHM']
    delta = 2*np.pi * (q['f21'] - q['f10'])['GHz']
    x = env.flattop(t0, len, w, amp, overshoot, overshoot_w)
    y = -alpha * env.deriv(x) / delta
    return x + 1j*y

# z rotations
def piPulseZ(q, t0):
    """Pi pulse using a gaussian envelope."""
    return rotPulseZ(q, t0, angle=np.pi)

def piHalfPulseZ(q, t0):
    """Pi/2 pulse using a gaussian envelope."""
    return rotPulseZ(q, t0, angle=np.pi/2)

def rotPulseZ(q, t0, angle=np.pi):
    """Rotation pulse using a gaussian envelope."""
    r = angle / np.pi
    return env.gaussian(t0, w=q['piFWHMZ'], amp=q['piAmpZ']*r)


# default pulse type is half-derivative
piPulse = piPulseHD
piHalfPulse = piHalfPulseHD
rotPulse = rotPulseHD


def spectroscopyPulse(q, t0, df=0):
    dt = q['spectroscopyLen']
    amp = q['spectroscopyAmp']
    return env.mix(env.flattop(t0, dt, w=q['piFWHM'], amp=amp), df)


def measurePulse(q, t0, state=1):
    """Add a measure pulse for the desired state.
    
    PARAMETERS
    q: Qubit dictionary.
    t0 - value [us]: Time to start the measure pulses.
    state - scalar: Which state's measure pulse to use.
    """
    return env.trapezoid(t0, 0, q['measureLenTop'], q['measureLenFall'], ml.getMultiLevels(q,'measureAmp',state))


def measurePulse2(q, t0):
    return env.trapezoid(t0, 0, q['measureLenTop2'], q['measureLenFall2'], q['measureAmp2'])


def readoutPulse(q, t0):
    dt = q['readoutLen']
    amp = power2amp(q['readout power'])
    df = q['readout frequency'] - q['readout fc']
    return env.mix(env.flattop(t0, dt, w=q['readoutWidth'], amp=amp), df)


def boostState(q, t0, state):
    """Excite the qubit to the desired state, concatenating pi pulses as needed.
    
    PARAMETERS
    q: Qubit dictionary.
    t0 - value [ns]: Time to start the pulses (center of first pi pulse).
    state - scalar: State to which qubit should be excited.
    """
    xypulse = env.NOTHING
    for midstate in range(state):
        xypulse = xypulse + mix(q, piPulse(q, t0+midstate*q['piLen'], state=(midstate+1)), state=(midstate+1))
    return xypulse
        


# sequence corrections

def correctCrosstalkZ(qubits):
    """Adjust the z-pulse sequences on all qubits for z-xtalk."""
    biases = [q.z for q in qubits]
    for q in qubits:
        coefs = list(q['calZpaXtalkInv'])
        q.z = sum(float(c) * bias for c, bias in zip(coefs, biases))

