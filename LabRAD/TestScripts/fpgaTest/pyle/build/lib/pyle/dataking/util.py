import labrad
from labrad.units import Unit,Value
mK = Unit('mK')
    
def loadDeviceType(sample ,deviceType, write_access=False):
    Devices=[]
    devices=[]
    deviceNames = sample['config']
    #First get writeable Devices
    for deviceName in deviceNames:
        if sample[deviceName]['_type'] == deviceType:
            Devices.append(sample[deviceName])
    #Now make the unwritable devices
    sample = sample.copy()
    for deviceName in deviceNames:
        if sample[deviceName]['_type'] == deviceType:
            devices.append(sample[deviceName])
    if write_access:
        return sample, devices, Devices
    else:
        return sample, devices

def loadQubits(sample, write_access=False):
    """Get local copies of the sample configuration stored in the registry.
    
    Returns the local sample config, and also extracts the individual
    qubit configurations, as specified by the sample['config'] list.  If
    write_access is True, also returns the qubit registry wrappers themselves,
    so that updates can be saved back into the registry.
    """
    Qubits = [sample[q] for q in sample['config']]  #RegistryWrappers
    sample = sample.copy()                          #AttrDict
    qubits = [sample[q] for q in sample['config']]  #AttrDicts
    
    # only return original qubit objects if requested
    if write_access:
        return sample, qubits, Qubits
    else:
        return sample, qubits

def loadDevices(sample, write_access=False):
    """Get local copies of the sample configuration stored in the registry.

    Returns the local sample config, and also extracts the individual
    device configurations, as specified by the sample['config'] list.  If
    write_access is True, also returns the qubit registry wrappers themselves,
    so that updates can be saved back into the registry.
    """
    devices={}
    Devices={}
    #The order of these lines is important, as we want devices to be assigned
    #after we make a copy of sample, whereas Devices is not a copy.
    for d in sample['config']:
        Devices[d]=sample[d]
    sample = sample.copy()
    for d in sample['config']:
        devices[d]=sample[d]

    if write_access:
        return sample, devices, Devices
    else:
        return sample, devices


def dcZero():
    for id in [2, 3, 5, 6]:
        board = 'DR Lab FastBias %d' % id
        for chan in ['A', 'B', 'C', 'D']:
            dcVoltage(board, chan, 0)


def dcVoltage(board, chan, voltage):
    with labrad.connect() as cxn:
        channels = [('b', ('FastBias', [board, chan]))]
        p = cxn.qubit_sequencer.packet()
        p.initialize([('dev', channels)])
        p.mem_start_timer()
        p.mem_bias([('b', 'dac1', voltage)])
        p.mem_stop_timer()
        p.build_sequence()
        p.run(30)
        p.send()


def getMixTemp(cxn, sample, measure=0):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    q, Q = qubits[measure], Qubits[measure]
    device_name = q['lakeshoreName']
    lr = cxn.lakeshore_ruox
    lr.select_device(device_name)
    temp = lr.temperatures()[0][0][mK]
    Q['temperature'] = temp
    return temp