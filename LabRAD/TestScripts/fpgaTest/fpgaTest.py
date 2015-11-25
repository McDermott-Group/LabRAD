# Author: Daniel Sank
# Created: 2010

# CHANGELOG
#
# 2012 Sep 26 - Jim Wenner
# Added documentation. Added adcAmpToVoltage, dacAmpToAdcAmp, dacAmpToVoltage to calibrate
# adc/dac wrt Volts. Code debugging.
#
# 2011 Feb 10 - Daniel Sank
# Added the instructions

# + STARTUP INSTRUCTIONS
# In order to use fpgaTest.py follow these instructions:
# 1. Make a shortcut to Ipython.
# 2. Set up the shortcut to run the conf.py file found in this directory
  # a. Right click on the shorcut and choose Properties.
  # b. In the Target field add "<labrad path>\scripts\fpgaTest\conf.py"
  #    The complete target path should be something like this:
  #    C:\Python26\python.exe C:\Python26\scripts\ipython <labrad path>\scripts\fpgaTest\conf.py
  # c. Set the Start In field to "<labradPath>\scripts\fpgaTest"
# 3. Make sure you are connected to the LabRAD system, either through Commando using Putty or by working on a computer that's on the Gbit network.
  # a. Make sure you have the direct ethernet and GHzFPGA servers running
# 4. Set up the LabRAD registry
  # a. Create a directory in the LabRAD registry called TestUser
  # b. Inside TestUser create
    # i. directory 'fpgaTest'
    # ii. key sample = ['fpgaTest']
  # c. Inside fpgaTest create
    # i. key config = [<boardName0>,<boardName1>,...].
    #    Board names must match the names you get when you call the list_devices() setting on the GHzFPGA server.
    # ii. directories <boardName0>, <boardName1>,...
  # d. Create waveform definition keys in the directory for each board.
    # i. DAC
      # I.      _id = <boardName>, must match name of directory
      # II.     signalAmplitude = [amplitudeDacA,amplitudeDacB],    eg  [0.5,0.3]
      # III.    signalDc = [dcLevelDacA,dcLevelDacB],               eg. [0.001,0.000]
      # IV.     signalFrequency = [freqDacA,freqDacB],              eg. [2.0 MHz, 5.0 MHz]
      # V.      signalPhase = [phaseDacA,phaseDacB],                eg. [0.0, 0.01] (this is in CYCLES)
      # VI.     signalTime = totalWaveformTime                      eg. 3.0 us
      #VII.     signalWaveform = [waveformDacA,waveformDacB]        eg. ['sine','square']
    # ii. ADC
      # I.      _id = <boardName>, must match name of directory
      # II.     demods = [(ch0Freq,ch0Phase,ch0CosAmp,ch0SinAmp),...(ch3Freq,ch3Phase,ch3CosAmp,ch3SinAmp)],
      #         eg. [(10.0 MHz, 0, 255, 255),...]
      # III.    filterFunc = (type,width)                           eg. ('gaussian',0.1)
      # IV.     filterStretchAt = index to use for the stretch ,    eg. 0
      # V.      filterStretchLen = length of filter stretch         eg. 0
      # Using 0 for filterAt and filterStretch gives you no stretch, use nonzero values to get stretch
# 5. Double click the shortcut you made in steps 1 and 2.
# 6. You now have an open python session. To use the functions in fpgaTest.py enter commands as follows:
# >> fpgaTest.functionName(s,fpga, <other arguments>) or fpgaTest.functionName(s,cxn, <other arguments>)
# The first argument "s" gives the program information about what boards you're
# running. The second argument essentially gives the program a LabRAD connection object
# so that it can make requests on the LabRAD system.
# inf. Read the source code to see what functions are available. Have fun :)
# 

#TODO
#
# In ADC scans, use all four available demod channels to make things run faster

import sys
import time
import numpy as np
from msvcrt import getch, kbhit
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit

import labrad
from labrad.units import Value

FPGA_SERVER = 'ghz_fpgas'
DATA_VAULT = 'data_vault'

import LabRAD.Servers.Instruments.GHzBoards.dac as dacModule
import LabRAD.Servers.Instruments.GHzBoards.adc as adcModule
import LabRAD.TestScripts.fpgaTest.pyle.pyle.dataking.util as dataUtil
import LabRAD.TestScripts.fpgaTest.pyle.pyle.util.sweeptools as st

from labrad.units import Value, GHz, MHz, Hz, ns, us

ADC_DEMOD_CHANNELS = 4
DAC_ZERO_PAD_LEN = 16
DAC_CHANNELS=2

def boardName2Info(boardName):
    """Gets the (number, type) of an FPGA board from the name"""
    pattern = '\d+' #match any number of digits
    p = re.compile(pattern)
    m = p.search(boardName)
    sp = m.span()
    boardNumber = int(boardName[sp[0]:sp[1]])
    if 'DAC' in boardName:
        boardType = 'DAC'
    elif 'ADC' in boardName:
        boardType = 'ADC'
    else:
        raise Exception('Board type of board %s not recognized' %boardName)
    return (boardNumber, boardType)
    
def makePlot(datasets, title, xlabel, ylabel):
    """Make a plot with multiple datasets

    datasets = [{'x':xData,'y':yData,'marker':markerString}]
    """
    plt.figure()
    plt.title(title)
    for data in datasets:
        if 'x' in data.keys():
            plt.plot(data['x'], data['y'], data['marker'])
        else:
            plt.plot(data['y'], data['marker'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
def loadDacsAdcs(sample):
    sample, devices = dataUtil.loadDevices(sample)
#    sample, devices = dataUtil.loadDevices(sample)
    dacs = []
    adcs = []
    for d in devices.keys():
        if 'DAC' in d:
            dacs.append(devices[d])
        elif 'ADC' in d:
            adcs.append(devices[d])
        else:
            raise Exception('Device must be DAC or ADC')
    return dacs, adcs
    
def saveData(cxn,folder,name,independents,dependents,data,params=None):
    dv = cxn[DATA_VAULT]
    dv.cd(folder,True)
    dv.new(name,independents,dependents)
    dv.add(data)
    if params is not None:
        for key in sorted(params):
            dv.add_parameter((key,params[key]))
    

#########################
## Communication check ##
#########################


def pingBoardDirect(de, port, boardName, verbose=False):
    """Use the direct ethernet server directly to ping a board
    Do not call this function is someone is taking data!!! You are bypassing
    the FPGA server and therefore can muck up synchronized operation. If you run
    this function
    PARAMETERS
    ----------
    de: server wrapper
        Object pointing to the direct ethernet server talking to the board you
        want to ping.
    port: number
        Port number, on the direct ethernet server, connected to your board.
    boardName: str
        The board's name, eg. 'Vince DAC 11'
        
    OUTPUT
    ----------
    (str - raw packet response from board, dict - parsed response)

    See the processReadback functions in adc.py and dac.py for format of
    parsed responses.
    """
    boardNumber, boardType = boardName2Info(boardName)
    if boardType == 'DAC':
        module = dacModule
    elif boardType == 'ADC':
        module = adcModule
    else:
        raise Exception('Board type not recognized')
    mac = module.macFor(boardNumber)
    ctxt = de.context()                     #Get a new context to work with
    try:
        p = de.packet(context=ctxt)
        p.connect(port, context=ctxt)           #Choose ethernet port
        p.require_source_mac(mac, context=ctxt) #Receive only packets from the board's mac address
        p.destination_mac(mac, context=ctxt)    #Set the destination mac of our outgoing packets
        p.listen(context=ctxt)                  #Start listening for packets
        boardPkt = module.regPing()
        p.write(boardPkt.tostring())            #Convert the board packet to a byte string and write it out over the wire
        p.timeout(1.0)
        p.read()
        result = p.send()                       #Send the direct ethernet server packet and get the result
        raw = result.read
        src, dst, eth, data = raw
        parsedInfo = module.processReadback(data)
        if verbose:
            print '\n'
            print 'Response from direct ethernet server: \n'
            print raw[3]
            print '\n'
            print 'Parsed response:'
            for key,value in parsedInfo.items():
                print key,value
    finally:
        de._cxn.manager.expire_context(context=ctxt)
    return (raw,parsedInfo)
    
    
def getBuildNumber(fpga, device):
    """Use the FPGA server to get a board build number.
    PARAMETERS
    ---------
    device - str: Name of device
    
    OUTPUT
    ---------
    str - board build number
    """
    fpga.select_device(device)
    buildNumber = fpga.build_number()
    return buildNumber
    
    
def hammarEthernet(de, port, boards, numRuns):
    """Send many register ping packets to the boards to check reliability of ethernet connection
    
    Packets for each board are sent sequentially, ie. not at the same time. This manes we are
    testing only the bare ethernet communication for individual channels and are not sensitive
    to packet collision type failure modes.
    """
    boardInfo = [boardName2Info(board) for board in boards]
    boardRegs = []
    ctxts = []
    macs = []
    modules = {'DAC': dacModule, 'ADC': adcModule}
    #Get a direct ethernet context, MAC, and register packet for each board
    for num, boardType in zip(boards, boardInfo):
        module = modules[boardType]
        boardRegs.append(module.regPing())
        ctxts.append(de.context())
        macs.append(module.macFor(num))
    #Set up connection to direct ethernet, one context per board
    for ctxt, mac in zip(ctxts, macs):
        p = de.packet(context=ctxt)
        p.timeout(1.0)
        p.connect(port, context=ctxt)                   #Choose ethernet port
        p.require_source_mac(mac, context=ctxt)         #Receive only packets from the board's mac address
        p.destination_mac(mac, context=ctxt)            #Set the destination mac of our outgoing packets
        p.listen(context=ctxt)                          #Start listening for packets
        p.send()
    #Ping the boards a lot
    for _ in range(numRuns):
        for regs, ctxt in zip(boardRegs, ctxts):
            p = de.packet(context=ctxt)
            p.write(regs.tostring(), context=ctxt)      #Convert the board packet to a byte string and write it out over the wire
            p.send()                                    #Send the direct ethernet server packet
    #Check to see that number of returned packets is as expected
    try:
        for ctxt in ctxts:
            p = de.packet(context=ctxt)
            p.timeout(10)
            p.collect(numRuns)
            p.send()
    except Exception:
        print Exception
        print 'Some boards dropped packets'
    finally:
        print 'Sequence done'
        print 'Check packet buffer. Number of packets should be %d' %numRuns
        print 'Then hit any key to continue'
        while kbhit():
            getch()
        getch()
        for ctxt in ctxts:
            de._cxn.manager.expire_context(context=ctxt)
    
    
def daisyCheck(sample, cxn, iterations, repsPerIteration, runTimers):
    """Set up a synchronized sequence on multiple boards and run it many
    times.
    
    This function is designed to make sure the daisychain is working. It
    can also be used to check whether the boards are running the correct
    number of times.
    
    INPUTS
    iterations, int: number of time to run run_sequence
    repsPerIteration, int: number of reps per run_sequence
    runTimers, bool: True will run timers on DAC boards. False will not.
    
    RETURNS
    Array of size (iterations x N) where N is number of boards returning
    timing data. Entry  ij is the number of times the jth board executed
    on the ith iteration.
    """
    fpga = cxn.ghz_fpgas
    for server in cxn.servers:
        if server.find('direct_ethernet') != -1:
            de = server
            break
    dacs, adcs = loadDacsAdcs(sample)
    for dac in dacs:
        dac['signalTime'] = 4.0*us
        sramLen = int(dac['signalTime']['ns'])
        #Construct the memory sequence for the DAC boards.
        #Note that the total memory sequence times are longer than the
        #packet transmission time of 10us. Also note that we include
        #an explicit delay in the DAC boards after running SRAM
        #to allow for the ADC boards to finish demodulating.
        memorySequence = [0x000000, # NoOp                                         1 cycle
                          0x800000, # SRAM start address                           1 cycle
                          0xA00000 + sramLen - 1, # SRAM end address               1 cycle
                          0xC00000, # call SRAM                                  100 cycles for 4us (300 assumed by fpga server)
                          0x300190] # Delay 400+1 cycles = 16us for ADC demod    401 cycles ... total 504 cycles
        
        if runTimers:
            memorySequence.extend([0x400000,  # start timer                        1 cycle
                                   0x300064,  # Delay 100+1 cycles, 4us          101 cycles
                                   0x400001]) # stop timer                         1 cycle ... 103 cycles
                                   
        memorySequence.extend([0xF00000]) # Branch back to start                   2 cycles... 2 cycles
        
        #Sequence times
        # With timers = 609 actual = 24.3us
        # No timers   = 506 actual = 20.2us
        #Note that the GHzFPGA server will always assume the SRAM is 12us long = 300 cycles
        #when estimating sequence length.
        
        dac['memory'] = memorySequence
        waves=makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
    for adc in adcs:
        adc['runMode'] = 'demodulate'
    daisychainList = [dac['_id'] for dac in dacs] + [adc['_id'] for adc in adcs]
    #Timing data includes the ADC always, and the DACs if we ran the timers
    timingOrderList = ['%s::%d' %(adc['_id'],chan) for adc in adcs for chan in range(ADC_DEMOD_CHANNELS)]
    if runTimers:
        timingOrderList.extend([dac['_id'] for dac in dacs])
    executions = np.array([])
    for iteration in range(iterations):
        print 'Running iteration %d' %iteration
        #Set up DAC
        [_sendDac(dac,fpga) for dac in dacs]
        #Set up ADC
        [_sendAdc(adc,fpga) for adc in adcs]
        #Set up board group
        fpga.daisy_chain(daisychainList)
        fpga.timing_order(timingOrderList)
        #Run the sequence
        try:
            result = fpga.run_sequence(repsPerIteration, True)
        except:
            print 'Error in sequence run;'
        #Whether or not there is an error, ping all boards to check how many times they executed
        finally:
            for board in dacs+adcs:
                name = board['_id']
                fpga.select_device(name)
                resp = fpga.execution_count()
                executions = np.hstack((executions, resp))
    return np.reshape(executions, (iterations,-1))


###################
## ADC CALIBRATE ##
###################

def calibrateAdc(fpga, device):
    """Set ADC PLL and recalibrate the AD chip.
    PARAMETERS
    ------------
    device - str: Device name, ie. 'Vince ADC 1'
    """
    if not device in fpga.list_adcs():
        raise Exception('Can only run calibrateAdc on ADC boards!')
    fpga.select_device(device)
    fpga.adc_bringup()

###############
## PLL CHECK ##
###############

def initializePll(fpga, device):
    """Reset DAC PLL"""
    fpga.select_device(device)
    fpga.pll_init()

###############
## DAC CHECK ##
###############

def dacSignal(sample, fpga, reps=30, loop=True, getTimingData=False, trigger=None):
    """Send SRAM sequences out of DAC boards.
    
    PARAMETERS
    --------------
    reps - int: number of times to repeat SRAM sequence. Only used when loop=False.
    
    loop - bool: If True, each board will loop through its SRAM sequence
    continuously until stopped (ie. told to do something else).  This
    uses fpga.dac_run_sram which is completely asynchronous execution,
    (ie. daisychain not used, boards not synched together).
    If False, each board will run through its SRAM a number of times
    determined by reps, and will then idle at the first SRAM value.
    (I don't know if this is the first physical SRAM address or the
    one given by the SRAM start address, ie. memory code 0x8...)
    This uses fpga.run_sequence which _does_ use the daisychain and
    therefore ensures that each board starts at the same time. However,
    boards with different SRAM lengths won't line up after the rep.
    
    getTimingData - bool: Whether or not to collect timing data from the
    boards. If true, the data is returned.
    
    trigger - iterable: sequence defining trigger output. For alternating
    on/off use trigger = np.array([1,0,1,0,...]) etc.
    1=trigger on, 0=trigger off
    
    OUTPUT
    ----------
    Array of timing results. See fpga server for details.
    """
    dacs,adcs = loadDacsAdcs(sample)
    
    for dac in dacs:
        #Set up DAC
        sramLen = int(dac['signalTime']['ns'])
        #memory Sequence
        memory = [
            0x000000, # NoOp
            0x800000, # SRAM start address
            0xA00000 + sramLen - 1, # SRAM end address
            0xC00000, # call SRAM
            0x300190, # Delay 400+1 clock cycles, ~16us
            0x400000, # start timer
            0x3000C8, # Delay 200+1 cycles, 8us
            0x400001, # stop timer
            0xF00000, # Branch back to start
        ]
        dac['memory']=memory
        
        #sram
        waves=makeDacWaveforms(dac)
        dac['sram'] = waves2sram(waves[0],waves[1])
        #Optional custom trigger
        if trigger is not None:
            dac['sram'] = dacTrigger(dac['sram'][:],trigger)
            
        _sendDac(dac,fpga)
    
    #Set up board group
    daisychainList=[dac['_id'] for dac in dacs]
    timingOrderList=[dac['_id'] for dac in dacs]
    fpga.daisy_chain(daisychainList)
    fpga.timing_order(timingOrderList)

    if loop:
        for dac in dacs:
            fpga.select_device(dac['_id'])
            fpga.dac_run_sram(dac['sram'], loop)
    else:
        if getTimingData:
            return fpga.run_sequence(reps, getTimingData)
        else:
            fpga.run_sequence(reps, getTimingData)
    
    
def dacSignalCorrected(sample, fpga, sbFreq=None, amp=None, trigger=None):
    """Run looping SRAM corrected by deconvolution server
    
    This is designed to give a calibration of uwave power vs DAC
    amplitude when the deconvolution server is in effect.
    """
    dacs,adcs = loadDacsAdcs(sample)
    
    for dac in dacs:
        #Set up DAC
        dac['signalWaveform'] = ['sine', 'sine']
        dac['signalDC'] = [0, 0]
        dac['signalTime'] = 6*us
        if sbFreq is not None:
            dac['signalFrequency'] = [sbFreq, sbFreq]
            dac['signalPhase'] = [0.0, 0.25 if sbFreq>0 else -0.15]
        if amp is not None:
            dac['signalAmp'] = [amp, amp]
        sramLen = int(dac['signalTime']['ns'])
        #Make sure the signal is periodic on the time window
        assert ((dac['signalFrequency'][0]*dac['signalTime']).value)%1 == 0
        #memory Sequence
        memory = [
            0x000000, # NoOp
            0x800000, # SRAM start address
            0xA00000 + sramLen - 1, # SRAM end address
            0xC00000, # call SRAM
            0x3000FF, # Delay 255+1 clock cycles 
            0xF00000, # Branch back to start
        ]
        dac['memory']=memory
        
        #sram
        waves=makeDacWaveforms(dac)
        #correct signal
        calibration = sample._cxn.dac_calibration
        p = calibration.packet()
        p.board(dac['_id'])
        p.frequency(dac['carrierFrequency'])
        p.loop(True)
        p.correct(zip(waves[0],waves[1]), key='corrected')
        result = p.send()
        correctedSRAM = result['corrected']
        #Pack data for fpga server
        sramI, sramQ = correctedSRAM
        sramI = sramI[500:2500] #Cut to 2us long sample
        sramQ = sramQ[500:2500]
        sramI = [long(i) for i in sramI]
        sramQ = [long(q) for q in sramQ]
        truncatedI=[y & 0x3FFF for y in sramI]
        truncatedQ=[y & 0x3FFF for y in sramQ]
        dacAData = truncatedI
        dacBData=[y<<14 for y in truncatedQ]
        sram=[dacAData[i]|dacBData[i] for i in range(len(dacAData))] #Combine DAC A and DAC B
        dac['sram'] = sram
        #Optional custom trigger
        if trigger is not None:
            dac['sram'] = dacTrigger(dac['sram'][:],trigger)
        _sendDac(dac,fpga)
    
    #Set up board group
    daisychainList=[dac['_id'] for dac in dacs]
    timingOrderList=[dac['_id'] for dac in dacs]
    fpga.daisy_chain(daisychainList)
    fpga.timing_order(timingOrderList)

    for dac in dacs:
        fpga.select_device(dac['_id'])
        fpga.dac_run_sram(dac['sram'], True)

 
##################
## AVERAGE MODE ##
##################

def runAverage(sample, fpga, plot=False):
    """Run the ADC board in average mode asynchronously (at a compeltely random time). No
    daisy chaining is used.
    
    For each ADC board, time traces are measured for the I and Q channels. Does NOT run
    the DAC boards!
    
    OUTPUT
    [(ADC0 I, ADC0 Q),(ADC1 I, ADC Q),...]
    """
    dacs, adcs = loadDacsAdcs(sample)
    
    if not adcs:
        raise Exception('No ADCs found, check registry settings')
    results=[]
    for adc in adcs:
        fpga.select_device(adc['_id'])
        adc['runMode']='average'
        _sendAdc(adc, fpga)
        ans = fpga.adc_run_average()
        results.append(ans)
        if plot:
            makePlot([{'y':ans[0],'marker':'.'},{'y':ans[1],'marker':'r.'}],
                      '%s average mode' %adc['_id'],
                      'Sample number','Value [bits]')
    return results
    
    
def trackPhase(sample, fpga, filename='U:\FPGAsof\phaseTrack_terminated45.txt'):
    fout = open(filename, "w")
    plt.figure()
    iteration=0
    while kbhit():
        getch()
    key=0
    while 1:
        if kbhit():
            while kbhit():
                key = getch()
        if key=='\x1b': #Escape key
            fout.close()
            break
        iteration+=1
        #Get new data
        result = runAverage(sample, fpga, plot=False)
        vals = result[0][22:24]
        #plt.plot(iteration,vals[0],'.',iteration,vals[1],'r.')
        fout.write('%d, %d \n' %(vals[0],vals[1]))
        fout.flush()
        print '%d, %d -- Press ESC and then wait to quit program' %(vals[0],vals[1])
        time.sleep(10)
    
    
def average(sample, cxn, reps, plot=False, save=True, name='ADC Average', folder=None):
    """
    Check ADC average functionality by averaging (summing) repeated
    waveforms generated by the DAC. Output on the plot is summed signal,
    but reps rounded up to nearest multiple of 30.
    """
    fpga = cxn[FPGA_SERVER]
    dacs, adcs = loadDacsAdcs(sample)
    if len(adcs)>1:
        raise Exception('Only one ADC allowed for average mode check.')
    if len(dacs)>1 and save:
        raise Exception('Only one DAC allowed for average mode check if saving enabled.')
    if not len(dacs) or not len(adcs):
        raise Exception('Must have at least one DAC and one ADC. Check registry setup.')
    adc=adcs[0]
    
    for dac in dacs:
        sramLen = int(dac['signalTime']['ns'])
        
        #DAC Memory sequence
        memory = [
            0x000000, # NoOp
            0x800000, # SRAM start address
            0xA00000 + sramLen - 1, # SRAM end address
            0xC00000, # call SRAM
            0x3186A0, # Delay 4ms to ensure average mode readback completes on A/D
            0x30C350, # Delay 2ms safety margin
            0x400000, # start timer
            0x400001, # stop timer
            0xF00000, # branch back to start
        ]

        #Set up DAC
        dac['memory']=memory
        waves = makeDacWaveforms(dac)
        dac['sram'] = waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
    
    #Set up ADC
    adc['runMode']='average'
    _sendAdc(adc, fpga)
    
    #Board group setups
    daisyChainList = [dac['_id'] for dac in dacs] + [adc['_id']]
    fpga.daisy_chain(daisyChainList)
    fpga.timing_order([adc['_id']])
    
    data = fpga.run_sequence(reps, True)
    I, Q = data[0]
    
    if plot:
        plt.figure()
        t = np.linspace(0, 2 * (np.size(I) - 1), np.size(I))
        plt.plot(t, I, '.')
        plt.plot(t, Q,'r.')
        plt.xlabel('Time [ns]')
        plt.ylabel('ADC Amplitude [ADC bits]')
        plt.show()
    if save:
        dac=dacs[0]
        if folder is None:
            folder = sample._dir
        params = dict(dac.items()+adc.items())
        params['reps'] = reps
        dataSave = np.array([2.0*np.array(range(len(I))),I,Q]).T
        print np.shape(dataSave)
        saveData(cxn,folder,name,[('Time','ns')],[('I','',''),('Q','','')],dataSave,params)
    return (I, Q)
    
    
def sumCheck(sample, cxn, repetitions=None, plot=False, collect=False,
            save=True):
    if repetitions is None:
        repetitions = [30, 60, 90]
    for elem in repetitions:
        if elem%30!=0:
            raise Exception('All reps must be multiples of 30.')
    data=[]
    markers = ['b.', 'r.', 'g.']
    for reps in repetitions:
        result = average(sample, cxn, reps, save=save)
        data.append(result)
    if plot:
        t = np.linspace(0, 2 * (np.size(data[0][0]) - 1),
                                np.size(data[0][0]))
        # Plot I waveform for each number of repetitions.
        plt.figure()
        for reps, dat, marker in zip(repetitions, data, markers):
            plt.plot(t, dat[0], marker,
                    label=str(reps) + ' reps', markersize=10)
        plt.legend()
        plt.title('ADC I Waveforms for Various Average Numbers')
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [ADC bits]')
        plt.grid()
        plt.show()

    if collect:
        return data

######################
## DEMODULATOR MODE ##
######################

def spectrum(sample, cxn, freqScan=st.r[-20:20:0.1,MHz], filter=None,
            dacFreq=None, reps=30, plot=False,
            save=True, name='Spectrum', folder=None):
    """ Scan the ADC demodulator frequency with a fixed DAC output tone
    
    Each channel of the DAC will put out a tone at fixed frequency (with
    a 0.25 cycle phase difference between each channel). These signals can
    either be sent directly into the I and Q ports of the ADC board, or
    they can used with an IQ mixer to generate a single sideband tone
    which will then be downconverted by another IQ mixer before going into
    the ADC board.

    The ADC collects this signal, and does the digital demodulation to DC
    at each frequency in freqScan. Thus, the ADC board acts like a
    spectrum analyzer, checking the signal generated by the DAC board at
    a range of frequencies.
    
    The DAC should be set with both channels putting out sinewaves at the
    same frequency but with a 0.25 phase difference between. For example,
    signalPhase = [0.0, 0.25]. Both amplitudes MUST be identical, or else
    you will see a spurious negative sideband peak.   
    
    """
    fpga = cxn[FPGA_SERVER]
    dacs, adcs = loadDacsAdcs(sample)
    assert len(dacs)==1, 'Only one DAC allowed'
    assert len(adcs)==1, 'Only one ADC allowed'
    dac=dacs[0]
    adc=adcs[0]
    #Default setups
    if dacFreq is not None:
        dac['signalFrequency']=[dacFreq,dacFreq]
    if filter is not None:
        adc['filterFunc']=filter    

    data=np.array([])
    freqs=[]
    print 'freqScan', freqScan
    for demodFreq in freqScan:
        freqs.append(demodFreq['MHz'])
        print 'Acquiring data at',demodFreq
        #Set up DAC
        sramLen=int(dac['signalTime']['ns'])
        dac['memory']=dacMemory(sramLen, 25000)
        waves=makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
        #Set up ADC
        adc['runMode']='demodulate'
        for chan in range(ADC_DEMOD_CHANNELS):
            modifyDemodChannel(adc, chan, freq=demodFreq)
        _sendAdc(adc,fpga)
        #Set up board group
        daisychainList = [dac['_id'],adc['_id']]
        timingOrderList = ['%s::%d' %(adc['_id'],chan) for chan in range(ADC_DEMOD_CHANNELS)]
        fpga.daisy_chain(daisychainList)
        fpga.timing_order(timingOrderList)
        #Run the sequence
        result = fpga.run_sequence(reps,True)
        # Parse/average the data
        mags=np.zeros(ADC_DEMOD_CHANNELS)
        for chan in range(ADC_DEMOD_CHANNELS):
            Is, Qs = result[chan]
            I, Q = np.mean(Is), np.mean(Qs)
            print 'chan', chan, 'I', I
            mag = (I**2)+(Q**2)
            mags[chan]=mag
        data = np.hstack((data,mags))
    data = np.reshape(data,(-1, ADC_DEMOD_CHANNELS))
    freqs=np.array(freqs)
    if plot:
        plt.figure()
        markers = ['b','r','g','k']
        for chan in range(ADC_DEMOD_CHANNELS):
            plt.semilogy(freqs,data[:,chan],markers[chan],label=str(adc['demods'][chan][2]))
            print 'freqs', freqs
            print 'data', data[:,chan]
        plt.legend(loc="upper left")
        plt.grid()
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Power [arb. units]')
        plt.show()
    if save:
        if folder is None:
            folder = sample._dir
        dependents = []
        dataSave = [freqs]
        for chan in range(ADC_DEMOD_CHANNELS):
            dependents += [('Power',str(adc['demods'][chan][2]),'')]
            dataSave += [data[:,chan]]
        params = dict(dac.items()+adc.items())
        params['reps'] = reps
        saveData(cxn,folder,name,[('Frequency','MHz')],dependents,np.array(dataSave).T,params)
        
    return np.vstack((freqs,data.T)).T
    
    
def sideband(sample, cxn, sourceFreq=None, freqScan=st.r[-20:20:0.5,MHz], filter=None, plot=False, 
             save=True, name='Sideband', folder=None):
    """ Checks differentiation between positive and negative sidebands.
    
    Runs spectrum. sourceFreq is (+/-) sideband frequency.
    
    Each channel of the DAC will put out a tone at fixed frequency (with
    a 0.25 cycle phase difference between each channel). These signals can
    either be sent directly into the I and Q ports of the ADC board, or
    they can used with an IQ mixer to generate a single sideband tone
    which will then be downconverted by another IQ mixer before going into
    the ADC board.

    The ADC collects this signal, and does the digital demodulation to DC
    at each frequency in freqScan. Thus, the ADC board acts like a
    spectrum analyzer, checking the signal generated by the DAC board at
    a range of frequencies.
    
    The DAC should be set with both channels putting out sinewaves at the
    same frequency but with a 0.25 phase difference between. For example,
    signalPhase = [0.0, 0.25]. Both amplitudes MUST be identical, or else
    you will see a spurious negative sideband peak.   
    
    """
    results=[]
    if sourceFreq is None:
        dacs, adcs = loadDacsAdcs(sample)
        dac=dacs[0]
        adc=adcs[0]
        sourceFreq = dac['signalFrequency'][0]
    if plot:
        plt.figure()
    if save:
        dependents = []
        dataSave = [[freqSideband[MHz] for freqSideband in freqScan]]
    for freq,marker in zip([-sourceFreq,sourceFreq],['b','r']):
        data = spectrum(sample, cxn, freqScan=freqScan, filter=filter,
                dacFreq=freq, save=save)
        results.append(data)
        if plot:
            plt.semilogy(data[:,0],data[:,1],marker,label=str(freq))
        if save:
            dependents += [('Power',str(freq),'')]
            dataSave += [data[:,1]]
    if plot:
        plt.legend(loc="upper center")
        plt.grid()
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Power [arb. units]')
        plt.show()
    if save:
        if folder is None:
            folder = sample._dir
        params = dict(dac.items()+adc.items())
        saveData(cxn,folder,name,[('Frequency','MHz')],dependents,np.array(dataSave).T,params)
    return results
    
def filterCompare(sample, cxn, filters, freqScan=st.r[-20:20:0.5,MHz],
        plot=False, save=True, name='Filter Compare', folder=None):
    """
    Plot spectra with different ADC filter functions applied.
    
    filters is of the form [(Filter1,Num1),(Filter2,Num2),...]. FilterN can be
    'square', 'gaussian', or 'hann'. NumN only matters if FilterN is 'gaussian',
    in which case NumN is the standard deviation.
    
    sweepPhases should be 0.25 apart.
    """
    dacs, adcs = loadDacsAdcs(sample)
    dac=dacs[0]
    adc=adcs[0]
    data=np.array([])
    results={}
    for filter in filters:
        data = spectrum(sample, cxn, freqScan=freqScan, filter=filter,
                save=save)
        data = data[:,:2]
        results[filter]=data
    if plot:
        plt.figure()
        markers=['.','r.']
        for i,(filter,data) in enumerate(results.items()):
            plt.semilogy(data[:,0], data[:,1], label=filter[0],
                linewidth=3)
        plt.grid()
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Power [arb. units]')
        plt.title('Spectra for different filter functions')
        plt.legend(loc='upper center')
        plt.show()
    if save:
        if folder is None:
            folder = sample._dir
        dependents = []
        dataSave = [[freqSideband[MHz] for freqSideband in freqScan]]
        for i,(filter,data) in enumerate(results.items()):
            dependents += [('Power',filter[0],'')]
            dataSave += [data[:,1]]
        params = dict(dac.items()+adc.items())
        saveData(cxn,folder,name,[('Frequency','MHz')],dependents,np.array(dataSave).T,params)
    return results
    
    
def phase(sample, cxn, sweepPhaseOf, freq=10.0*MHz, reps=30, phaseScan=np.linspace(-0.495,0.495,100),
            dacAmp = None, plot=False, noisy=True, save=True, name='Phase of ', folder=None):
    """
    Sweep DAC or ADC phases over a full cycle.
    
    sweepPhaseOf is either 'DAC' or 'ADC' and determines which phase is swept. sweepPhases
    should differ by 0.25. phaseScan must lie within (-0.5,0.5), the acceptable phase range
    of the ADC. For determining the center of the circle, all points should be equidistant,
    including the endpoints.
    """
    fpga = cxn[FPGA_SERVER]
    if sweepPhaseOf not in ['DAC','ADC']:
        raise Exception('Can only sweep phase of DAC or ADC')
    #Load devices from registry
    dacs, adcs = loadDacsAdcs(sample)
    if len(dacs)>1 or len(adcs)>1:
        raise Exception('Only one DAC board and one ADC board allowed.')
    dac=dacs[0]
    adc=adcs[0]
    
    #Set signal parameters
    dacStartPhase = dac['signalPhase']
    if dacAmp is not None:
        dac['signalAmplitude']=[Value(dacAmp),Value(dacAmp)]
    
    Is = np.array([])
    Qs = np.array([])
    for phase in phaseScan:
        if noisy:
            print 'Acquiring data at ',phase
        #Set up DAC
        sramLen = int(dac['signalTime']['ns'])
        memory = dacMemory(sramLen, 400)
        dac['signalFrequency']=[freq,freq]
        if sweepPhaseOf=='DAC':
            dac['signalPhase'] = [dacStartPhase[0]+phase,dacStartPhase[1]+phase]
        dac['memory']=memory
        waves = makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
        #Set up ADC
        if sweepPhaseOf=='ADC':
            modifyDemodChannel(adc, 0, freq=freq, phase=phase)
        else:
            modifyDemodChannel(adc, 0, freq=freq)
        adc['runMode']='demodulate'
        _sendAdc(adc,fpga)
        #Set up synchronized run
        fpga.timing_order(['%s::0' %adc['_id']])
        fpga.daisy_chain([dac['_id'],adc['_id']])
        result = fpga.run_sequence(reps,True)
        I=np.mean(result[0][0])
        Q=np.mean(result[0][1])
        Is = np.hstack((Is,I))
        Qs = np.hstack((Qs,Q))
    if plot:
        plt.figure()
        plt.plot(Is, Qs, '.', markersize=15)
        plt.plot(np.mean(Is), np.mean(Qs), 'g.', markersize=15)
        plt.plot(0, 0, 'r+', markersize=15)
        plt.title('ADC IQ data, %s phase swept over one cycle' %sweepPhaseOf)
        plt.xlabel('I [ADC bits]')
        plt.ylabel('Q [ADC bits]')
        plt.grid()
        plt.axis('equal')
        plt.show()
    if save:
        if folder is None:
            folder = sample._dir
        params = dict(dac.items()+adc.items())
        params['reps'] = reps
        name += sweepPhaseOf
        saveData(cxn,folder,name,[('I','')],[('Q','','')],
                np.vstack((np.array([Is,Qs]).T,np.array([np.mean(Is),np.mean(Qs)]))),params)
    print Is
    return (Is,Qs)
    
    
def rings(sample, fpga, amplitudes):
    data=[]
    for amplitude in amplitudes:
        result = phase(sample, fpga, 'DAC', dacAmp=amplitude)
        data.append(result)
    plt.figure()
    plt.show()
    for dat in data:
        plt.plot(dat[0],dat[1],'.',markersize=18)
        plt.grid()

########################
## SIGNAL CALIBRATION ##
########################

def dacAmpToAdcAverage(sample, cxn, freq=10.0*MHz, reps=30, ampScan=np.append(np.logspace(-4,-1,91),np.linspace(0.11,1.0,90)),
            plot=False, plotAll=False, noisy=True, save=True, name='DAC Amp to ADC Average', folder=None):
    """Measures ADC amplitude (in average mode) vs DAC amplitude.
    
    The DAC must be set with both channels putting out sinewaves at the
    same frequency but with a 0.25 phase difference between. For example,
    signalPhase = [0.0, 0.25]. Both amplitudes MUST be identical.
    """
    fpga = cxn[FPGA_SERVER]
    #Load devices from registry
    dacs, adcs = loadDacsAdcs(sample)
    if len(dacs)>1 or len(adcs)>1:
        raise Exception('Only one DAC board and one ADC board allowed.')
    dac=dacs[0]
    adc=adcs[0]
    
    #Set signal parameters
    dac['signalFrequency']=[freq,freq]
    def sineFunc(x, amp, phase, offset):
        """amp*sin(2*pi*(x*freq+phase))+offset
        amp: amplitude
        freq: frequency (cycles per unit x)
        phase: phase (cycles)
        offset: DC offset
        """
        return amp*np.sin(2*np.pi*(x*freq[GHz]+phase))+offset
    
    Iclicks = np.array([])
    Qclicks = np.array([])
    for amp in ampScan:
        if noisy:
            print 'Acquiring data at ',amp
        #Set up DAC
        dac['signalAmplitude']=[Value(amp),Value(amp)]
        sramLen = int(dac['signalTime']['ns'])
        memory = dacMemory(sramLen, 400)
        dac['memory']=memory
        waves = makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
        #Set up ADC
        adc['runMode']='average'
        _sendAdc(adc, fpga)
        #Set up synchronized run
        fpga.timing_order([adc['_id']])
        fpga.daisy_chain([dac['_id'],adc['_id']])
        data = fpga.run_sequence(reps, True)
        I, Q = data[0]
        Icut = np.array(I[(sramLen/4):(sramLen/2)])/(1.0*reps)
        Qcut = np.array(Q[(sramLen/4):(sramLen/2)])/(1.0*reps)
        times = 2*np.array(range(len(Icut)))
        Ifit = curve_fit(sineFunc, times, Icut, [100,0,0])[0]
        Qfit = curve_fit(sineFunc, times, Qcut, [100,0,0])[0]
        Ifit = curve_fit(sineFunc, times, Icut, Ifit)[0]
        Qfit = curve_fit(sineFunc, times, Qcut, Qfit)[0]
        if plotAll:
            plt.figure()
            plt.plot(times,Icut,'b.',markersize=20,label='I')
            plt.plot(times,Qcut,'r.',markersize=20,label='Q')
            plt.plot(times,sineFunc(times,Ifit[0],Ifit[1],Ifit[2]),'b')
            plt.plot(times,sineFunc(times,Qfit[0],Qfit[1],Qfit[2]),'r')
            plt.title('ADC Demod Amplitude vs. Time')
            plt.xlabel('Time [ns]')
            plt.ylabel('ADC Amplitude (ADC bits)')
            plt.grid()
            plt.show()
        Iclicks = np.hstack((Iclicks,np.abs(Ifit[0])))
        Qclicks = np.hstack((Qclicks,np.abs(Qfit[0])))

    if plot:
        plt.figure()
        plt.plot(ampScan,Iclicks,'.',markersize=10,label='I')
        plt.plot(ampScan,Qclicks,'.',markersize=10,label='Q')
        plt.title('ADC Average Amplitude vs. DAC Input Amplitude')
        plt.xlabel('DAC Amplitude')
        plt.ylabel('ADC Amplitude (ADC bits)')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.loglog(ampScan,Iclicks,'.',markersize=10,label='I')
        plt.loglog(ampScan,Qclicks,'.',markersize=10,label='Q')
        plt.title('ADC Average Amplitude vs. DAC Input Amplitude')
        plt.xlabel('DAC Amplitude')
        plt.ylabel('ADC Amplitude (ADC bits)')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    if save:
        if folder is None:
            folder = sample._dir
        params = dict(dac.items()+adc.items())
        params['reps'] = reps
        saveData(cxn,folder,name,[('DAC Amplitude','')],[('Amplitude I','ADC Average','Clicks'),('Amplitude Q','ADC Average','Clicks')],np.array([ampScan,Iclicks,Qclicks]).T,params)
    return (ampScan,Iclicks,Qclicks)
        
def adcAmpToVoltage(sample, cxn, scope=None, freq=10.0*MHz, reps=30,
        ampScan=np.append(np.logspace(-4,-1,91),np.linspace(0.11,1.0,90)),
        plot=False, noisy=True, save=True, folder=None):
    dacAmps, adcAmps = dacAmpToAdcAmp(sample, cxn, freq=freq, reps=reps,
            ampScan=ampScan, plot=plot, noisy=noisy, save=save, 
            folder=folder)
    if scope is not None:
        dacAmpToVoltage(sample,cxn,scope,freq=freq,reps=1,ampScan=[1],plot=False,plotAll=plot,noisy=noisy,save=save,folder=folder)
        dacAmpsDC, voltsDC = dacAmpToVoltage(sample,cxn,scope,freq=freq,reps=1,ampScan=ampScan,plot=plot,plotAll=False,noisy=noisy,coupling='DC',save=save,folder=folder)
        dacAmpsAC, voltsAC = dacAmpToVoltage(sample,cxn,scope,freq=freq,reps=1,ampScan=ampScan,plot=plot,plotAll=False,noisy=noisy,coupling='AC',save=save,folder=folder)
        
    if plot and scope is not None:
        plt.figure()
        plt.plot(voltsDC,adcAmps,'.',markersize=20)
        plt.title('ADC Demod Amplitude vs. DC-Coupled Voltage')
        plt.xlabel('Voltage [V]')
        plt.ylabel('ADC Amplitude')
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.loglog(voltsDC,adcAmps,'.',markersize=20)
        plt.title('ADC Demod Amplitude vs. DC-Coupled Voltage')
        plt.xlabel('Voltage [V]')
        plt.ylabel('ADC Amplitude')
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(voltsAC,adcAmps,'.',markersize=20)
        plt.title('ADC Demod Amplitude vs. AC-Coupled Voltage')
        plt.xlabel('Voltage [V]')
        plt.ylabel('ADC Amplitude')
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.loglog(voltsAC,adcAmps,'.',markersize=20)
        plt.title('ADC Demod Amplitude vs. AC-Coupled Voltage')
        plt.xlabel('Voltage [V]')
        plt.ylabel('ADC Amplitude')
        plt.grid()
        plt.show()

def dacAmpToAdcAmp(sample, cxn, freq=10.0*MHz, reps=30, ampScan=np.append(np.logspace(-4,-1,91),np.linspace(0.11,1.0,90)),
            plot=False, noisy=True, save=True, name='DAC Amp to ADC Demod', folder=None):
    """Measures ADC amplitude vs DAC amplitude.
    
    The DAC must be set with both channels putting out sinewaves at the
    same frequency but with a 0.25 phase difference between. For example,
    signalPhase = [0.0, 0.25]. Both amplitudes MUST be identical.
    """
    fpga = cxn[FPGA_SERVER]
    #Load devices from registry
    dacs, adcs = loadDacsAdcs(sample)
    if len(dacs)>1 or len(adcs)>1:
        raise Exception('Only one DAC board and one ADC board allowed.')
    dac=dacs[0]
    adc=adcs[0]
    
    #Set signal parameters
    dac['signalFrequency']=[freq,freq]
    
    mags = np.array([])
    for amp in ampScan:
        if noisy:
            print 'Acquiring data at ',amp
        #Set up DAC
        dac['signalAmplitude']=[Value(amp),Value(amp)]
        sramLen = int(dac['signalTime']['ns'])
        memory = dacMemory(sramLen, 400)
        dac['memory']=memory
        waves = makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
        #Set up ADC
        modifyDemodChannel(adc, 0, freq=freq)
        adc['runMode']='demodulate'
        _sendAdc(adc,fpga)
        #Set up synchronized run
        fpga.timing_order(['%s::0' %adc['_id']])
        fpga.daisy_chain([dac['_id'],adc['_id']])
        result = fpga.run_sequence(reps,True)
        I=np.mean(result[0][0])
        Q=np.mean(result[0][1])
        mags = np.hstack((mags,np.sqrt(I**2+Q**2)))

    if plot:
        plt.figure()
        plt.plot(ampScan,mags, '.', markersize=10)
        plt.title('ADC Demod Amplitude vs. DAC Input Amplitude')
        plt.xlabel('DAC Amplitude [DAC units]')
        plt.ylabel('ADC Amplitude [ADC bits]')
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.loglog(ampScan,mags, '.', markersize=10)
        plt.title('ADC Demod Amplitude vs. DAC Input Amplitude')
        plt.xlabel('DAC Amplitude [DAC units]')
        plt.ylabel('ADC Amplitude [ADC bits]')
        plt.grid()
        plt.show()
    if save:
        if folder is None:
            folder = sample._dir
        params = dict(dac.items()+adc.items())
        params['reps'] = reps
        saveData(cxn,folder,name,[('DAC Amplitude','')],[('ADC Demod Amplitude','','')],np.array([ampScan,mags]).T,params)
    return (ampScan,mags)
    
    
def dacAmpToVoltage(sample, cxn, scope, freq=10.0*MHz, reps=1, ampScan=np.append(np.logspace(-4,-1,91),np.linspace(0.11,1.0,90)),
            differential=True, coupling='DC', plot=False, plotAll=False, noisy=True,
            save=True, name='DAC Amp to Voltage', folder=None):
    """Measures voltage vs DAC amplitude.
    
    scope is the oscilloscope server. The oscilloscope must be selected
    prior to running this program.
    
    The DAC must be set with both channels putting out sinewaves at the
    same frequency but with a 0.25 phase difference between. For example,
    signalPhase = [0.0, 0.25]. Both amplitudes MUST be identical.  
    
    Since the two channels are putting out identical sine waves differing
    by 1/4 cycle (e.g., sin and cos), (A^2+B^2)^.5 is the magnitude. A and
    B are both differential signals. Hence, all 4 traces must be hooked up.
   
    If differential=False, uses only one output from both diff amps.
    """
    fpga = cxn[FPGA_SERVER]
    #Load devices from registry
    dacs, adcs = loadDacsAdcs(sample)
    if len(dacs)>1:
        raise Exception('Only one DAC board allowed.')
    dac=dacs[0]
    adc=adcs[0]
    
    #Set signal parameters
    dac['signalFrequency']=[freq,freq]
    def sineFunc(x, amp, phase, offset):
        """amp*sin(2*pi*(x*freq+phase))+offset
        amp: amplitude
        freq: frequency (cycles per unit x)
        phase: phase (cycles)
        offset: DC offset
        """
        return amp*np.sin(2*np.pi*(x*freq[GHz]+phase))+offset
    
    #Hook up the scope.
    print('Check that the following are connected to the scope.')
    print('Use equal length cables')
    if differential:
        print('   DAC A+ -> CH 1')
        print('   DAC A- -> CH 2')
        print('   DAC B+ -> CH 3')
        print('   DAC B- -> CH 4')
        channels = [1,2,3,4]
    else:
        print('   DAC A -> CH 1')
        print('   DAC B -> CH 2')
        channels = [1,2]
    print('Hit enter when done.')
    raw_input()
    
    #Set up the scope.
    scope.trigger_mode('AUTO')
    time.sleep(5) #Giving time to reset screen.
    for channel in channels:
        scope.channelOnOff(channel, 'ON')
        scope.invert(channel, 0)
        scope.termination(channel, 50)
        scope.coupling(channel, 'DC')
        scope.position(channel, 0)
        scope.coupling(channel,coupling)
        
    scope.trigger_slope('RISE')
    scope.trigger_level(0)
    scope.trigger_channel('CH1')
    scope.trigger_mode('NORM')
    
    horizPosition = 0
    horizScale = .3/(freq[Hz])
    scope.horiz_position(horizPosition)
    scope.horiz_scale(horizScale)
    
    Iamps = np.array([])
    Qamps = np.array([])
    for amp in ampScan:
        if noisy:
            print 'Acquiring data at ',amp
            
        #Set up DAC
        dac['signalAmplitude']=[Value(amp),Value(amp)]
        sramLen = int(dac['signalTime']['ns'])
        memory = dacMemory(sramLen, 400)
        dac['memory']=memory
        waves = makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
        
        #Set up Scope
        scale = amp*.75/4 #0.75V = 1 DAC Amp, Each channel p-p = 2 divs
        for channel in channels:
            scope.scale(channel, scale)
        time.sleep(1)
            
        #Set up synchronized run and read out from scope
        result = fpga.run_sequence(reps,True)
        scopeData = [scope.get_trace(ch) for ch in channels]
        times = np.array([onetime[ns] for onetime in scopeData[0][0]])
        volts = np.array([[volt['V'] for volt in trace[1]] for trace in scopeData])
        
        if differential:
            I = volts[0]-volts[1]
            Q = volts[2]-volts[3]
        else:
            I = volts[0]
            Q = volts[1]
        Ifit = curve_fit(sineFunc, times, I, [100,0,0])[0]
        Qfit = curve_fit(sineFunc, times, Q, [100,0,0])[0]
        Ifit = curve_fit(sineFunc, times, I, Ifit)[0]
        Qfit = curve_fit(sineFunc, times, Q, Qfit)[0]
        Iamps = np.hstack((Iamps,np.abs(Ifit[0])))
        Qamps = np.hstack((Qamps,np.abs(Qfit[0])))
        
        if plotAll:
            plt.figure()
            plt.show()
            plt.plot(times,I,'b.',markersize=20,label='I')
            plt.plot(times,Q,'r.',markersize=20,label='Q')
            plt.plot(times,sineFunc(times,Ifit[0],Ifit[1],Ifit[2]),'b')
            plt.plot(times,sineFunc(times,Qfit[0],Qfit[1],Qfit[2]),'r')
            plt.title('DAC Voltage Vs Time, Amp %f, %s-Coupled' %(amp,coupling))
            plt.xlabel('Time [ns]')
            plt.ylabel('Voltage [V]')

    if plot:
        plt.figure()
        plt.plot(ampScan,Iamps,'.',markersize=20,label='I')
        plt.plot(ampScan,Qamps,'.',markersize=20,label='Q')
        plt.show()
        plt.legend(loc='lower right')
        plt.title('Voltage vs. DAC Input Amplitude, %s-Coupled' %coupling)
        plt.xlabel('DAC Amplitude')
        plt.ylabel('Voltage [V]')
        plt.grid()
        
        plt.figure()
        plt.loglog(ampScan,Iamps,'.',markersize=20,label='I')
        plt.loglog(ampScan,Qamps,'.',markersize=20,label='Q')
        plt.show()
        plt.legend(loc='lower right')
        plt.title('Voltage vs. DAC Input Amplitude, %s-Coupled' %coupling)
        plt.xlabel('DAC Amplitude')
        plt.ylabel('Voltage [V]')
        plt.grid()
    if save:
        if folder is None:
            folder = sample._dir
        params = dict(dac.items()+adc.items())
        params['reps'] = reps
        params['coupling'] = coupling
        params['differential'] = differential
        saveData(cxn,folder,name,[('DAC Amplitude','')],[('Amplitude','I','V'),('Amplitude','Q','V')],np.array([ampScan,Iamps,Qamps]).T,params)
    return (ampScan,Iamps,Qamps)
    
    
def dacAmpToPower(sample, cxn, sa, iqFreq, freq=10.0*MHz, sidebands=[-1,0,1,2],
            ampScan=np.append(np.logspace(-4,-1,91),np.linspace(0.11,1.0,90)),
            plot=False, noisy=True,
            save=True, name='DAC Amp to Power', folder=None):
    """Measures voltage vs DAC amplitude.
    
    sa is the spectrum analyzer server. The spectrum analyzer must be
    selected prior to running this program. iqFreq is the CW frequency
    input into the IQ mixer.
    
    The DAC must be set with both channels putting out sinewaves at the
    same frequency but with a 0.25 phase difference between. For example,
    signalPhase = [0.0, 0.25]. Both amplitudes MUST be identical.  
    
    Since the two channels are putting out identical sine waves differing
    by 1/4 cycle (e.g., sin and cos), (A^2+B^2)^.5 is the magnitude. A and
    B are both differential signals. Hence, all 4 traces must be hooked up.
   
    If differential=False, uses only one output from both diff amps.
    """
    fpga = cxn[FPGA_SERVER]
    #Load devices from registry
    dacs, adcs = loadDacsAdcs(sample)
    if len(dacs)>1:
        raise Exception('Only one DAC board allowed.')
    dac=dacs[0]
    
    #Set signal parameters
    dac['signalFrequency']=[freq,freq]
    dac['signalWaveform']=['sine','sine']
    
    #Set up the spectrum analyzer
    sa.set_span(0)
    
    powers = []
    freqScans = [iqFreq+sideband*freq for sideband in sidebands]
    for amp in ampScan:
        if noisy:
            print 'Acquiring data at ',amp
            
        #Set up DAC
        dac['signalAmplitude']=[Value(amp),Value(amp)]
        sramLen = int(dac['signalTime']['ns'])
        memory = dacMemory(sramLen, 255)
        dac['memory']=memory
        waves = makeDacWaveforms(dac)
        dac['sram']=waves2sram(waves[0],waves[1])
        _sendDac(dac,fpga)
            
        #Set up loop run
        fpga.select_device(dac['_id'])
        fpga.dac_run_sram(dac['sram'], True)
        
        #Set up and read Spectrum Analyzer
        power=[]
        for freqScan in freqScans:
            sa.set_center_frequency(freqScan[MHz])
            time.sleep(5)
            power.append(np.mean(sa.get_trace()[2]))
        powers.append(power)
    powers = np.array(powers).T
    
    if plot:
        plt.figure()
        plt.show()
        for freqScan,power,color in zip(freqScans,powers,['b','r','g','k']):
            plt.plot(ampScan,power,'.',markersize=20,label=str(freqScan),color=color)
        plt.title('Power vs. DAC Input Amplitude')
        plt.xlabel('DAC Amplitude')
        plt.ylabel('Power [dBm]')
        plt.grid()
    if save:
        if folder is None:
            folder = sample._dir
        params = dict(dac.items())
        params['CW Frequency'] = iqFreq
        for freqScan,power in zip(freqScans,powers):
            saveData(cxn,folder,name+' '+str(freqScan),[('DAC Amplitude',''),],[('Power','','dBm')],np.array([ampScan,power]).T,params)
    return (ampScan,powers)

######################################################
## Fun functions to write messages in the IQ plane. ##
######################################################

def art(sample, fpga, points, freq=10.0*MHz):
    
    Is = np.array([])
    Qs = np.array([])
    polars =[]
    xs = np.array(zip(*points)[0],dtype='float64')
    ys = np.array(zip(*points)[1],dtype='float64')
    rs = np.sqrt((xs**2)+(ys**2))
    scale = max(rs)
    xs = 0.25*xs/scale
    ys = 0.25*ys/scale
    rs = np.sqrt((xs**2)+(ys**2))
    cycles = np.arctan2(ys,xs)/(2.0*np.pi)

    for i in range(len(rs)):        
        I,Q = phase(sample, fpga, 'ADC', freq=freq, reps=30, phaseScan=np.array([cycles[i]]),
                    dacAmp = rs[i], plot=False, noisy=False)
        Is = np.hstack((Is,I))
        Qs = np.hstack((Qs,Q))
    plt.figure()
    plt.show()
    plt.plot(Is,Qs,'.',markersize=18)
    plt.grid()
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Our ADC rocks!')
    
    
def message(sample,fpga,message):
    raise Exception('message defined in two places - here and before letters. FIX ME!!!')
    points=[]
    for i,let in enumerate(message):
        x0 = CENTERS[i][0]
        y0 = CENTERS[i][1]
        points.extend(letter(let,x0,y0))
    art(sample, fpga, points)

#######################
## Utility functions ##
#######################

def modifyDemodChannel(adc, chan, freq=None, phase=None, ampI=None, ampQ=None):
    demod = adc['demods'][chan]
    Freq, Phase, AmpI, AmpQ = demod
    if freq is not None:
        Freq = freq
    if phase is not None:
        Phase = phase
    if ampI is not None:
        AmpI = ampI
    if ampQ is not None:
        AmpQ = ampQ
    adc['demods'][chan] = (Freq, Phase, AmpI, AmpQ)
    
def _sendDac(dac, server):
    p = server.packet()
    p.select_device(dac['_id'])
    p.memory(dac['memory'])
    p.sram(dac['sram'])
    p.send()

def _sendAdc(adc, server):
    p = server.packet()
    p.select_device(adc['_id'])
    p.start_delay(0)
    p.adc_run_mode(adc['runMode'])
    p.adc_filter_func(filterBytes(adc), adc['filterStretchLen'], adc['filterStretchAt'])
    #Set up demod channels
    for chan in range(ADC_DEMOD_CHANNELS):
        frequency, phase, ampSin, ampCos = adc['demods'][chan]
        dPhi = frequency2dPhi(frequency)
        phi0 = cycles2phi0(phase)
        p.adc_demod_phase(chan, dPhi, phi0)
        p.adc_trig_magnitude(chan, ampSin, ampCos)
    p.send()

def filterBytes(adc):
    filterFunc = adc['filterFunc'][0]
    sigma = adc['filterFunc'][1]
    filterLen = 4096
    filt = np.zeros(filterLen,dtype='<u1')
    if filterFunc=='square':
        filt = filt+128
    elif filterFunc=='gaussian':
        env = np.linspace(-0.5,0.5,filterLen)
        env = np.floor(128*np.exp(-((env/(2*sigma))**2)))
        filt = np.array(filt+env,dtype='<u1')
    elif filterFunc=='hann':
        env = np.linspace(0,filterLen-1,filterLen)
        env = np.floor(128*np.sin(np.pi*env/(filterLen-1))**2)
        filt = np.array(filt+env,dtype='<u1')
    else:
        raise Exception('Filter function %s not recognized' %filterFunc)
    return filt.tostring()

def makeDacWaveforms(dac):
    T=dac['signalTime']['ns']
    freqs=[]
    amps=[]
    phases=[]
    dc=[]
    print'time', T
    for i in range(DAC_CHANNELS):
        print dac['signalFrequency'][0]['GHz']
        freqs.append(dac['signalFrequency'][i]['GHz'])
        amps.append(dac['signalAmplitude'][i].value)
        phases.append(dac['signalPhase'][i].value)
        dc.append(dac['signalDc'][i].value)
    for amp in amps:
        if amp>1.0:
            raise Exception('DAC amplitude cannot exceed 1.0')
            
    t=np.linspace(0,int(T),int(T)) #data type is float64
    
    def makeImpulse(amp,dc):
        wave = np.zeros(len(t))
        wave[301:308] = np.zeros(7)+amp
        return wave
    def makeSine(amp, freq, phase, dc=0.0):
        stuff = amp * np.sin(2*np.pi*((t*freq)+phase))+dc
        stuff[0:DAC_ZERO_PAD_LEN] = dc
        return stuff
    def makeSquare(amp, freq, phase):
        return np.mod(np.floor([(((t*freq)+phase)*2) for t in range(int(T))]),2.0)-0.5
    def makeSawtooth(amp, freq, phase):
        return np.array([np.mod((float(t)*freq),1.0) for t in range(T)])
    def makeConstant(amp):
        return np.array([amp for t in range(T)])
    waves=[]
    for i in range(DAC_CHANNELS):
        if dac['signalWaveform'][i]=='impulse':
            wave=makeImpulse(amps[i], dc[i])
        elif dac['signalWaveform'][i]=='sine':
            wave=makeSine(amps[i], freqs[i], phases[i], dc[i])
        elif dac['signalWaveform'][i]=='square':
            wave=makeSquare(amps[i], freqs[i], phases[i])
        elif dac['signalWaveform'][i]=='sawtooth':
            wave=makeSawtooth(amps[i], freqs[i], phases[i])
        elif dac['signalWaveform'][i]=='constant':
            wave=makeConstant(amps[i])
        else:
            raise Exception('Waveform type %s not recognized' %dac['signalWaveform'][i])
        waves.append(wave)
    return waves
    
def waves2sram(waveA, waveB):
    """Construct sram sequence for a list of waveforms"""
    if not len(waveA)==len(waveB):
        raise Exception('Lengths of DAC A and DAC B waveforms must be equal.')
    for q in np.hstack((waveA,waveB)):
        if q>1.0: raise Exception('Wave amplitude cannot exceed 1')
    dataA=[long(np.floor(0x1FFF*y)) for y in waveA]  #Multiply wave by full scale of DAC. DAC is 14 bit 2's compliment,
    dataB=[long(np.floor(0x1FFF*y)) for y in waveB]  #so full scale is 13 bits, ie. 1 1111 1111 1111 = 1FFF
    truncatedA=[y & 0x3FFF for y in dataA]          #Chop off everything except lowest 14 bits.
    truncatedB=[y & 0x3FFF for y in dataB]
    dacAData=truncatedA
    dacBData=[y<<14 for y in truncatedB]            #Shift DAC B data by 14 bits.
    sram=[dacAData[i]|dacBData[i] for i in range(len(dacAData))] #Combine DAC A and DAC B
    #for i in range(DAC_ZERO_PAD_LEN):               #Zero pad beginning of sequence. Do this because board idles at beginning of sram sequence.
    #    sram[i]=0                                   #This is now being done at time of waveform construction
    sram[16]|=0xF0000000                            #Add trigger pulse near beginning of sequence.
    sram[17]|=0xF0000000                            #Add trigger pulse near beginning of sequence.
    return sram
    
def frequency2dPhi(frequency):
    """Get trig table address step size for a demod frequency. Assumes sample rate is 500MHz on ADC"""
    
    assert isinstance(frequency,Value), 'data must be instance of labrad.units.Value with frequency units'
    assert frequency.isCompatible('Hz'), 'data must be instance of labrad.units.Value with frequency units'
    dPhi = int(np.floor((frequency/Value(7629.0,'Hz')).inBaseUnits().value))
    return dPhi

def cycles2phi0(phase):
    return int(phase*(2**16))

def dacMemory(sramLen,delayAfterSram):
    memory = [
            0x000000, # NoOp
            0x800000, # SRAM start address
            0xA00000 + sramLen - 1, # SRAM end address
            0xC00000, # call SRAM
            0x300000 + delayAfterSram,
            0x400000, # start timer
            0x30000F, # Delay
            0x400001, # stop timer
            0xF00000, # Branch back to start
        ]
    return memory

def dacTrigger(sram, trig):
    for i,t in enumerate(trig):
        sram[i]&=0x0FFFFFFF
        if t == 1:
            sram[i] |= 0xF0000000
    return sram
        
#######################
## Playing around  ####
#######################

CENTERS_THANKYOU = [(-950,500),
           (-450,500),
           (0,500),
           (500,500),
           (950,500),
           (-450,-500),
           (0,-500),
           (450,-500)
           ]

CENTERS_MARTINISGROUP = [(-1450,500),
                         (-950,500),
                         (-450,500),
                         (0,500),
                         (400,500),
                         (750,500),
                         (1100,500),
                         (1450,500),
                         (-950,-500),
                         (-450,-500),
                         (0,-500),
                         (450,-500),
                         (950,-500)
                         ]
def line(xi,yi,xf,yf,N):
    assert N>1, 'Must have at least two points for a line'
    vect = np.array([xf-xi,yf-yi])
    len = np.sqrt((vect[0]**2)+(vect[1]**2))
    vect = vect/len
    ds = len/(N-1) #Ensures we get the end points
    points=[]
    for i in range(N):
        points.append((xi+i*ds*vect[0],yi+i*ds*vect[1]))
    return points

def message(string,centers):
    raise Exception('message defined in two places - here and after art. FIX ME!!!')
    points=[]
    for i,let in enumerate(string):
        x0 = centers[i][0]
        y0 = centers[i][1]
        points.extend(letter(let,x0,y0))
    return points

def letter(letter,x0,y0):
    letter=letter.upper()
    if letter=='A':
        return letterA(x0,y0)
    elif letter=='G':
        return letterG(x0,y0)
    elif letter=='H':
        return letterH(x0,y0)
    elif letter=='I':
        return letterI(x0,y0)
    elif letter=='K':
        return letterK(x0,y0)
    elif letter=='M':
        return letterM(x0,y0)
    elif letter=='N':
        return letterN(x0,y0)
    elif letter=='O':
        return letterO(x0,y0)
    elif letter=='P':
        return letterP(x0,y0)
    elif letter=='R':
        return letterR(x0,y0)
    elif letter=='S':
        return letterS(x0,y0)
    elif letter=='T':
        return letterT(x0,y0)
    elif letter=='U':
        return letterU(x0,y0)
    elif letter=='Y':
        return letterY(x0,y0)

    else:
        print 'letter: ',letter
        raise Exception('Letter not implemented')

def letterA(x0,y0):
    line1 = line(x0-150,y0-200,x0-150,y0+100, 4)
    line2 = line(x0+150,y0-200,x0+150,y0+100, 4)
    line3 = line(x0-50,y0,x0+50,y0, 2)
    line4 = line(x0-50,y0+200,x0+50,y0+200, 2)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.extend(line4)
    return letter

def letterG(x0,y0):
    line1 = line(x0-150,y0+100,x0-150,y0-100, 3)
    line2 = line(x0-50,y0+200,x0+150,y0+200, 3)
    line3 = line(x0-50,y0-200,x0+50,y0-200, 2)
    line4 = line(x0+50,y0,x0+150,y0-100, 2)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.extend(line4)
    return letter

def letterH(x0,y0):
    line1 = line(x0-100,y0+200,x0-100,y0-200,5)
    line2 = line(x0+100,y0+200,x0+100,y0-200,5)
    letter = line1
    letter.extend(line2)
    letter.append((x0,y0))
    return letter

def letterI(x0,y0):
    letter = line(x0,y0+200,x0,y0-200, 5)
    return letter

def letterK(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0-200, 5)
    line2 = line(x0-50,y0,x0+150,y0+200, 3)
    line3 = line(x0-50,y0,x0+150,y0-200, 3)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    return letter

def letterM(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0-200, 5)
    line2 = line(x0-50,y0+100,x0,y0, 2)
    line3 = line(x0+150,y0+200,x0+150,y0-200, 5)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.append((x0+50,y0+100))
    return letter
    
def letterN(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0-200, 5)
    line2 = line(x0-150,y0+200,x0+150,y0-200, 5)
    line3 = line(x0+150,y0-200,x0+150,y0+200, 5)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    return letter

def letterO(x0,y0):
    line1 = line(x0-100,y0+100,x0-100,y0-100, 3)
    line2 = line(x0+100,y0+100,x0+100,y0-100, 3)
    letter = line1
    letter.extend(line2)
    letter.append((x0,y0+200))
    letter.append((x0,y0-200))
    return letter

def letterP(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0-200, 5)
    line2 = line(x0-50,y0+200,x0+50,y0+200, 2)
    line3 = line(x0-50,y0,x0+50,y0, 2)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.append((x0+150,y0+100))
    return letter

def letterR(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0-200, 5)
    line2 = line(x0-50,y0+200,x0+50,y0+200, 2)
    line3 = line(x0-50,y0,x0+50,y0, 2)
    line4 = line(x0+50,y0-100,x0+150,y0-200, 2)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.extend(line4)
    letter.append((x0+150,y0+100))
    return letter

def letterS(x0,y0):
    line1 = line(x0+150,y0+100,x0+50,y0+200, 3)
    line2 = line(x0+50,y0+200,x0-50,y0+200, 2)
    line3 = line(x0-50,y0+200,x0-150,y0+100, 3)
    line4 = line(x0-150,y0+100,x0+150,y0-100, 5)
    line5 = line(x0+150,y0-100,x0+50,y0-200, 3)
    line6 = line(x0+50,y0-200,x0-50,y0-200, 2)
    line7 = line(x0-50,y0-200,x0-150,y0-100, 3)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.extend(line4)
    letter.extend(line5)
    letter.extend(line6)
    letter.extend(line7)
    return letter

def letterT(x0,y0):
    line1 = line(x0-200,y0+200,x0+200,y0+200, 5)
    line2 = line(x0,y0+100,x0,y0-200, 4)
    letter = line1
    letter.extend(line2)
    return letter

def letterU(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0-100, 4)
    line2 = line(x0+150,y0+200,x0+150,y0-100, 4)
    line3 = line(x0-50,y0-200,x0+50,y0-200, 2)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    return letter

def letterY(x0,y0):
    line1 = line(x0-150,y0+200,x0-150,y0+100, 2)
    line2 = line(x0+150,y0+200,x0+150,y0+100, 2)
    line3 = line(x0-50,y0,x0+50,y0, 2)
    line4 = line(x0,y0-100,x0,y0-200, 2)
    letter = line1
    letter.extend(line2)
    letter.extend(line3)
    letter.extend(line4)
    return letter
