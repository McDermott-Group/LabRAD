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

NA_TYPE = '8730ET'
    
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
    
    
    
    
def run(cxn):
    
    data_folder = "Z:\mcdermott-group\Data\Syracuse Qubits\ADR Cooldown 070715\MH034"
    
    if not os.path.exists(data_folder):
            try:
                os.makedirs(data_folder)
            except:
                raise Exception('Could not create base path! Is AFS on?')
    
    filename = 'FreqvPower'
    
    #frequency

    
    nF = 801
    
    startF = 4.5 * GHz
    stopF = 5. * GHz
    
    freq = np.linspace(startF, stopF, nF)
    
    #power
    nP = 41
    
    power = np.linspace(-60, -20, nP)*dBm; 
    
    data = np.zeros((nF, nP))
    
    #averaging
    nAvg = 150
    
    na_server = cxn.agilent_8720et_network_analyzer()
    dev = na_server.list_devices()
    na_server.select_device(dev[0][0])
    
    #reduce output power out of caution
    na_server.power(-80*dBm)
    
    #set to transmission measurement
    na_server.measurement_setup('T')
    na_server.display_trace('LOGMAG')
    #na_server.
    na_server.start_frequency(startF)
    na_server.stop_frequency(stopF)
    na_server.sweep_points(nF)
    na_server.average_points(nAvg)
    na_server.average_mode(True)
   
    
    
    data = np.empty((nP,nF))
    
    for idx in xrange(nP):
        na_server.power(power[idx])
        data[idx,:] = na_server.get_trace()
        time.sleep(0.1)
        print power[idx]
        
    matSave(data_folder, filename, freq, power, data)
    
    
if __name__ == '__main__':
    cxn = labrad.connect()
    run(cxn)
    
        
    
    
    
    