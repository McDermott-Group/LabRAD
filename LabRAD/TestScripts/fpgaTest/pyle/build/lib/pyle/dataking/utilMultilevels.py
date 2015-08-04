# In the registry directory for a single qubit, we have certain important
# parameters such as pi-pulse amplitude, pi-pulse length, and measure
# amplitude. These parameters are state dependent; the amplitude for a pi
# pulse from the zero to one state will not be the same as the amplitude
# for a pi pulse from the one to two state. We store the parameters for
# different levels as separate keys. For example, the different pi amplitudes
# show up in the registry like this:
#
# piAmp
# piAmp2
# piAmp3
#
# This is nice when you're looking at the registry editor, but it's annoying
# for writing scripts where you want to be able to loop over pi amplitudes
# for different levels. To deal with this problem, you want some kind of list
# that contains all of the pi amplitudes, i.e. [piAmp, piAmp2, piAmp3,...]
# The code we've written here builds this list, and other such lists.
#
# For example, when you run setMultiKeys(qubit,max_state) then qubit (which
# is a python dictionary) will acquire a new entry called 'multiLevels' which
# is itself a dictionary containing a key 'piAmp' with value [piAmp,piAmp2,piAmp3].
# Therefore, the list you want to iterate over in your script is just
# qubit['multiLevels']['piAmp']. See the code for setMultiKeys for details.


def saveKeyNumber(key,state):
    """Create key name for higher state pulses for saving into registry.
    
    Inputs the registry key name and the state. Outputs the corresponding
    registry key referring to that state. Not valid for frequencies.
    
    Examples: setKeyNumber('piAmp',1) returns 'piAmp'
              setKeyNumber('piAmp',3) returns 'piAmp3'
    """
    statenum = str(state) if state>1 else ''
    newkey = key + statenum
    return newkey


def getMultiLevels(q,key,state):
    """Get value from local registry for higher state pulses.
    
    Inputs the registry key name (piLen,frequency,measureAmp) and the state.
    Outputs the corresponding value for that state.
    """
    if state<1:
        raise Exception('Dude, our qubits do not have negative levels')
    if ('multiLevels' not in q.keys()) or (q['multiLevels']['setFlag'] is False) or ('setFlag' not in q['multiLevels'].keys()):
        setMultiKeys(q,1)
        q['multiLevels']['setFlag'] = False
    return q['multiLevels'][key][state-1]

def setMultiLevels(q,key,val,state):
    """Set input value in local registry for higher state pulses.
    
    Inputs the registry key name (piLen or measureAmp), the value
    to be written, and the state for the value.
    """
    if state<1:
        raise Exception('Dude, our qubits do not have negative levels')
    if ('multiLevels' not in q.keys()) or (q['multiLevels']['setFlag'] is False):
        setMultiKeys(q,1)
        q['multiLevels']['setFlag'] = False
    q['multiLevels'][key][state-1] = val
    

def setMultiKeys(q,max_state):
    """Create lists of quantities for higher state pulses.
    
    Inputs the registry dictionary for the qubits and the maximum state which
    will be reached (PiAmp,f,measureAmp must be defined up to, and including,
    this state). Adds registry key multiLevels, which is a dictionary with keys
    frequency, measureAmp, and piAmp with values as a list of the appropriate
    items up to max_state.
    """
    PI_AMP_LEVELS = ['piAmp','piAmp2','piAmp3','piAmp4','piAmp5','piAmp6']
    FREQ_LEVELS = ['f10','f21','f32','f43','f54','f65']
    MPA_LEVELS = ['measureAmp','measureAmp2','measureAmp3','measureAmp4','measureAmp5','measureAmp6']
    
    if not q.has_key('multiLevels'):
        q['multiLevels']={}
    
    q['multiLevels']['setFlag']=True
    q['multiLevels']['frequency']=[q[key] for key in FREQ_LEVELS[:max_state]]
    q['multiLevels']['measureAmp']=[q[key] for key in MPA_LEVELS[:max_state]]
    q['multiLevels']['piAmp']=[q[key] for key in PI_AMP_LEVELS[:max_state]]
    
def setMultiKey(device,paramName,max_state):
    if not device.has_key('multiLevels'):
        device['multiLevels']={}
    device['multiLevels'][paramName] = [device[key] for key in [paramName+str(n) for n in range(max_state+1)[1:]]]
    
    
    