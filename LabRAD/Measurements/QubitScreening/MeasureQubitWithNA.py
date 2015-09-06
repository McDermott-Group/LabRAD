#MeasureQubitWithNA.py
#2015 Guilhem Ribeill

#Script to measure a qubit with a network analyzer

import numpy as np
import scipy.io as sio
import os
import time

import labrad
from labrad.units import (us, ns, V, GHz, MHz, rad, dB, dBm,
                          DACUnits, PreAmpTimeCounts)

#NA_TYPE = '8720ET'
NA_TYPE = 'N5230A'
    
def matSave(path, name, frequency, power, data):

    #which contents are files?
    onlyfiles = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    name = name.replace(" ","_")
    #which files start off with 'ExperimentName_'?
    files = [f.split('.')[0] for f in onlyfiles if f[:len(name)+1]==name+'_']
    
    #get file numbers and increment, or create first file if none in folder
    nums = [int(f[-3:]) for f in files if f[-3:].isdigit()]
    if nums==[]:
        num = '000'
        fname = name + '_' + num + '.mat'
    else:
        num = ("%03d" % (max(nums)+1,))
        fname = name + '_' + num + '.mat'
    
    filePath = os.path.join(path,fname)
    
    saveDict = {'Frequency': frequency,
                'Power': power,
                'Data':data}
    sio.savemat(filePath, saveDict)
    
    
def Setup8720ET(cxn):

    na_server = cxn.agilent_8720et_network_analyzer()
    dev = na_server.list_devices()
    na_server.select_device(dev[0][0])
    
    #reduce output power out of caution
    na_server.source_power(-80*dBm)
    
    #set to transmission measurement
    na_server.measurement_setup('T')
    na_server.display_trace('LOGMAG')
    
    return na_server
    
def SetupN5230A(cxn):

    na_server = cxn.agilent_5230a_network_analyzer()
    dev = na_server.list_devices()
    na_server.select_device(dev[0][0])
    
    na_server.source_power(-80*dBm)
    #na_server.measurement_setup('S43')
    #na_server.sweep_type('LIN')
    #na_server.if_bandwidth(0.05*MHz)
    
    return na_server
        
    
def run(cxn):
    
    data_folder = "Z:\mcdermott-group\Data\Syracuse Qubits\ADR Cooldown 090115\MH048B"
    
    if not os.path.exists(data_folder):
            try:
                os.makedirs(data_folder)
            except:
                raise Exception('Could not create base path! Is AFS on?')
    
    filename = 'FreqvPower'
    
    #frequency

    
    nF = 801
    
    centerF = 4.862*GHz
    span = 0.03*GHz
    
    freq = np.linspace(centerF['Hz']-span['Hz']/2., centerF['Hz']+span['Hz']/2., nF)
    
    #power
    nP = 41
    
    power = np.linspace(-50, -10, nP); 
    
    data = np.zeros((nF, nP))
    
    #averaging
    nAvg = 200
    
    if NA_TYPE == '8720ET':
        na_server = Setup8720ET(cxn)
    
    elif NA_TYPE == 'N5230A':
        na_server = SetupN5230A(cxn)
        
    else:
        print 'Unknown NA type'
        return
        
    na_server.start_frequency(centerF-span/2.)
    na_server.stop_frequency(centerF+span/2)
    na_server.sweep_points(nF)
    na_server.average_points(nAvg)
    na_server.average_mode(True)
   
    
    
    data = np.empty((nP,nF))
    
    for idx in xrange(nP):
        na_server.source_power(power[idx]*dBm)
        time.sleep(3)
        data[idx,:] = na_server.get_trace()

        print power[idx]
        
    matSave(data_folder, filename, freq, power, data)
    
    
if __name__ == '__main__':
    cxn = labrad.connect()
    run(cxn)
    
        
    
    
    
    