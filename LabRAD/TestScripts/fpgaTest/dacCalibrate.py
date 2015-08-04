# Copyright (C) 2012 Daniel Ssank
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

import fpgaTest
import fpgaTestUtil as util
from pyle import datasaver

import numpy as np
from labrad.units import Unit,Value

import time

Hz,MHz,GHz = (Unit(ss) for ss in ['Hz','MHz','GHz'])

DATAVAULT_PATH = ['','GHzDAC Calibration']

def makeDataset(cxn, board, name, indeps, deps, params=None):
    if params is None:
        params = []
    dv = cxn.data_vault
    dv.cd(DATAVAULT_PATH+[board])
    dv.new(name, indeps, deps)
    dv.add_parameters(params)
    
def diffAmpSingleOutVsClicks(sample, cxn, board, amps = None):
    """Measure the single ended voltage from a GHz DAC driving a Diff Amp as a
    function of DAC clicks. The deconvolution server is bypassed, so here a DAC
    amplitude of 1 is the actual full scale of the DAC.
    """
    #Default values
    if amps is None:
        amps = np.logspace(-4,0,20)
    dacs, adcs = util.loadDacsAdcs(sample)
    dac = [d for d in dacs if d['_id']==board][0]
    print 'Calibrating output voltage on %s' %board
    #Data vault setup
    dv = cxn.data_vault
    indeps = [('DAC amplitude', 'Fraction of full scale')]
    deps = [('Amplitude', 'DAC A', 'V'), ('Amplitude', 'DAC B', 'V')]
    makeDataset(cxn, board, 'DAC+DiffAmp single end voltage vs. clicks', indeps, deps)
    #Set up oscilloscope
    scope = cxn[dac['oscilloscopeServer']]
    scope.select_device(dac['oscilloscopeId'])
    scope.reset()
    scope.horiz_scale(Value(20,'ns'))
    for ch in [1,2]:
        scope.termination(ch, 50)
        scope.measure_type(ch, 'RMS')
        scope.coupling(ch, 'AC')
    #scope.measure_source(1, 'CH1')
    #scope.measure_source(2, 'CH2')
    _ = raw_input('Set measure source 1 to CH1 and source 2 to CH2. Then press ENTER')
    for amp in amps:
        print 'measuring at amp = %f' %amp
        dac['signalAmplitude'] = [Value(amp,''), Value(amp,'')]
        fpgaTest.dacSignal(sample, cxn.ghz_fpgas, reps=30, loop=True, getTimingData=False, trigger=None, dacs=[dac])
        scope.scale(1, np.sqrt(2)*Value(260,'mV')*amp/3.0)
        scope.scale(2, np.sqrt(2)*Value(260,'mV')*amp/3.0)
        time.sleep(3.0)
        #TODO: intelligently wait for the scope to acquire averages
        #TODO: acquire confidence of RMS value measured on scope
        rmsVal1 = scope.measure_value(1)
        rmsVal2 = scope.measure_value(2)
        dv.add(np.array([amp, np.sqrt(2)*rmsVal1['V'], np.sqrt(2)*rmsVal2['V']]))
        
        
def powerVsDacAmp(sample, cxn, board, sbFreq, carrierFreqs, amps):
    #load information from registry
    dacs, adcs =  util.loadDacsAdcs(sample)
    dac = [d for d in dacs if d['_id']==board][0]
    print dac['_id']
    print dac.keys()
    #Data vault setup
    indeps = [('Carrier Frequency', 'GHz'),('DAC amplitude', '')]
    deps = [('Power', '', 'dBm')]
    params = {'carrierFreq':dac['carrierFrequency']}
    #Name servers
    dv = cxn.data_vault
    sa = cxn.spectrum_analyzer_server
    fpga = cxn.ghz_fpgas
    uwSource = cxn[dac['uwSourceServer']]
    #Set up spectrum analyzer
    sa.select_device(dac['spectrumAnalyzerId'])
    sa.gpib_write(':POW:RF:ATT 0dB;:AVER:STAT OFF;:FREQ:SPAN 0Hz;:BAND 300Hz;'+ \
                  ':INIT:CONT OFF;:SYST:PORT:IFVS:ENAB OFF;:SENS:SWE:POIN 101')
    #Set up microwave source
    uwSource.select_device(dac['uwSourceId'])
    uwSource.amplitude(dac['uwSourcePower'])
    uwSource.output(True)
    #TODO: Make sure frequencies are all multiples of 0.5MHz
    #Do not go below 10MHz
    #Set up dataset
    makeDataset(cxn, board, 'PowerCal', indeps, deps, [(k,v) for k,v in params.items()])
    for carrierFreq in carrierFreqs:
        m = 1.0 if sbFreq['Hz']>0 else -1.0
        uwSource.frequency(carrierFreq)
        for amp in amps:
            amp = Value(amp)
            #Make a corrected output signal
            fpgaTest.dacSignalCorrected(sample, fpga, sbFreq, amp, dacs=[dac])
            #Set spectrum analyzer to look at desired sideband frequency
            sa.set_center_frequency(carrierFreq+(m*sbFreq))
            power_dBm = sa.gpib_query('*TRG;*OPC?;:TRAC:MATH:MEAN? TRACE1')
            power_dBm = float(power_dBm[2:])
            print 'carrierFreq: %s, amp: %s, power: %s dBm' %(carrierFreq, amp, power_dBm)
            dv.add(np.array([carrierFreq['GHz'], amp.value, power_dBm]))
