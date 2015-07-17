# Read two JPMs for correlation experiments.

import sys
import numpy as np

import labrad as lr

import DoubleJPMCorrelationExpt as djpm

G = float(10**9)
M = float(10**6)

# Connect to LabRAD.
cxn = lr.connect()

# List of the experiment resources. Simply uncomment/comment the devices that should be used/unused.
# However, 'Resource': 'LabRAD Server' should be never left out.
Resources = [   { # This is a required resource.
                 'Resource': 'LabRAD Server', 
                 'Server Name': 'ghz_fpgas'
                },
                { # List all parameters that specify waveforms here.
                    'Resource': 'Waveform Parameters',
                    'Variables': [
                                    'Init Time', 'Bias Time', 'Measure Time',
                                    'JPM A Bias Voltage', 'JPM A Fast Pulse Time', 'JPM A Fast Pulse Amplitude', 'JPM A Fast Pulse Width',
                                    'JPM B Bias Voltage', 'JPM B Fast Pulse Time', 'JPM B Fast Pulse Amplitude', 'JPM B Fast Pulse Width',
                                    'JPM A to JPM B Fast Pulse Delay', 'RF to JPM A Fast Pulse Delay',
                                    'RF Sideband Frequency', 'RF Time', 'RF Amplitude'
                                 ]
                },
                { # DACs are converted to a simple ordered list internally based on 'List Index' value.
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 3',
                    'List Index': 0,
                    'DAC Settings': {
                                        'DAC A': 'JPM A Fast Pulse',
                                        'DAC B': 'JPM B Fast Pulse',
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '1.0'
                                    },
                    'Variables': []
                },
                {
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 4',
                    'List Index': 1,
                    'DAC Settings': {   
                                        'DAC A': 'RF Q',
                                        'DAC B': 'RF I'
                                    },
                    'Variables': []
                },
                { # GPIB RF Generator
                    'Resource': 'RF Generator',
                    'GPIB Address': 'GPIB0::19',
                    'Variables': ['RF Power', 'RF Frequency']
                },
                { # Lab Brick Attenuator
                    'Resource': 'Lab Brick',
                    'Serial Number': 7033,
                    'Variables': ['RF Attenuation']
                },
                # { # SIM Voltage Source
                    # 'Resource': 'SIM',
                    # 'GPIB Address': 'GPIB::26',
                    # 'SIM Slot': 3,
                    # 'Variables': ['DC Bias']
                # },
                { # External readings.
                    'Resource': 'Manual Record',
                    'Variables': ['Temperature']
                },
                { # Extra experiment parameters.
                    'Resource': 'Software Parameters',
                    'Variables': ['Reps', 'Actual Reps', 'Threshold'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': '051215A-D10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Matched JPM Photon Counting\Leiden DR 2015-06-01 - JPMs 050115B-J5 and 051215A',
            'Experiment Name': 'RFA1D',
            'Comments': '051215A-D10 = JPM A; JPM-B = None. The RF excitation through the calibration line and the cold relays. 3 dB splitter at the RF output (to a DAC and an ADC boards).' 
           }
 
# Experiment Variables
ExptVars = {'Reps': 1000, # should not exceed ~50,000

            'RF Frequency': 4.9*G, # Hz
            'RF Power': 13, # dBm at RF generator output, should be in [-110, 13] range
            'RF Attenuation': 32, # dB, should be in (0, 63] range
                        
            'RF Sideband Frequency': 30*M, # Hz
            'RF Time': 2000, # nanoseconds
            'RF Amplitude': 0.5, # DAC units

            # 'DC Bias': 0, # DC Rack units

            'Init Time': 500, # microseconds
            'Bias Time': 100, # microseconds
            'Measure Time': 50, # microseconds

            'JPM A Bias Voltage': 0.198, # FastBias DAC units
            'JPM A Fast Pulse Time': 10, # nanoseconds
            'JPM A Fast Pulse Amplitude': 0.311, # DAC units
            'JPM A Fast Pulse Width': 0, #nanoseconds
                  
            'JPM B Bias Voltage': 0.4, # FastBias DAC units
            'JPM B Fast Pulse Time': 0, # nanoseconds
            'JPM B Fast Pulse Amplitude': 0, # DAC units
            'JPM B Fast Pulse Width': 0, # nanoseconds
            
            # Both JPM A and JPM B fast pulses should appear within the RF Pulse, i.e.
            # they should only start after the beginning of the RF Pulse and finish before the end of the RF Pulse.
            'RF to JPM A Fast Pulse Delay': 1000, # nanoseconds
            'JPM A to JPM B Fast Pulse Delay': 0, # nanoseconds
 
            'Threshold': 100,   # Preamp Time Counts
            'Temperature': 12.2 # mK
           }

with djpm.DoubleJPMCorrelation(cxn) as run:

    run.SetExperiment(ExptInfo, Resources, ExptVars)

    #p = run.RunOnce(Histogram=True)
    #print p

    # run.Sweep('JPM A Bias Voltage', np.linspace(0.19, 0.21, 101), 
                # Save=False, PrintData=['Pa'], PlotData=['Pa'],
                # Dependencies=[['Pa', 'JPM A Detection Time']])
    
    run.Sweep([['JPM A Bias Voltage'], ['JPM B Bias Voltage']],
                 [[np.linspace(0, .25, 101)], [np.linspace(0, .25, 101)]], 
                 Save=False, PrintData=['Pa', 'Pb'], PlotData=['Pa', 'Pb'],
                 Dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # run.Sweep(['JPM A Fast Pulse Amplitude', 'JPM A Bias Voltage'],
                 # [np.linspace(0, 1, 21), np.linspace(0, .2, 501)], 
                 # Save=True, PrintData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
    
    #run.Sweep('JPM A Fast Pulse Amplitude', np.linspace(.29, .4, 101), 
    #            Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
    
    # fpa = np.linspace(0.47, .5, 51)
    # run.Sweep([['JPM A Fast Pulse Amplitude'], ['JPM B Fast Pulse Amplitude']],
                 # [[fpa], [fpa]], 
                 # Save=True, PrintData=['Pa', 'Pb', 'P11'], PlotData=['Pa', 'Pb', 'P11'],
                 # Dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])

    # fpa_A = np.linspace(0.13, .15, 21)
    # fpa_B = np.linspace(0.47, .50, 21)
    # run.Sweep(['JPM A Fast Pulse Amplitude', 'JPM B Fast Pulse Amplitude'], [fpa_A, fpa_B], 
                 # Save=True, PrintData=['Pa', 'Pb', 'P11'])                
 
    # fpt = np.linspace(0, 200, 201)
    # run.Sweep([['JPM A Fast Pulse Time'], ['JPM B Fast Pulse Time']],
                 # [[fpt], [fpt]], 
                 # Save=True, PrintData=['Pa', 'Pb'], PlotData=['Pa', 'Pb'],
                 # Dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
                 
    #run.Sweep(['JPM A Fast Pulse Amplitude', 'JPM A Fast Pulse Time'], [np.linspace(0.30, .34, 21), np.linspace(0, 400, 51)], 
                 # Save=True, PrintData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
    
    # run.Variable('RF Attenuation', 63)
    # run.Variable('RF Frequency', 20*G)
    # run.Variable('RF Power', -110)
    # run.Sweep(['JPM A Fast Pulse Amplitude', 'JPM A Fast Pulse Time'], [np.linspace(0.30, .34, 21), np.linspace(0, 400, 51)], 
                 # Save=True, PrintData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])

    # run.Variable('JPM A Fast Pulse Amplitude', 0.056)
    # run.Variable('JPM B Fast Pulse Amplitude' ,0.075)
    # fpt = np.linspace(0, 1000, 1001)
    # run.Sweep([['JPM A Fast Pulse Time'], ['JPM B Fast Pulse Time']],
                 # [[fpt], [fpt]], 
                 # Save=True, PrintData=['Pa', 'Pb'], PlotData=['Pa', 'Pb'],
                 # Dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
                 
    # fpd = np.linspace(-500, 500, 201)
    # run.Sweep('JPM A to JPM B Fast Pulse Delay', fpd, 
                 # Save=True, PrintData=['Pa', 'Pb', 'P11'], PlotData=['Pa', 'Pb', 'P11'])
                 
    #run.Sweep('RF Frequency', np.linspace(3*G, 7*G, 201), 
    #            Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
                 
    # rff = np.linspace(.5*G, 6.5*G, 601)
    # run.Sweep([['RF Frequency'], ['RF Frequency']],
                 # [[rff], [rff]], 
                 # Save=True, PrintData=['Pa', 'Pb'], PlotData=['Pa', 'Pb'],
                 # Dependencies=[['Pa', 'JPM A Detection Time', 'JPM A Detection Time Std Dev'], 
                               # ['Pb', 'JPM B Detection Time', 'JPM B Detection Time Std Dev']])

    # rff = np.linspace(2*G, 8*G, 401)
    # rfa = np.linspace(43, 63, 11)
    # run.Sweep(['RF Attenuation', 'RF Frequency'], [rfa, rff], 
                # Save=True, PrintData=['Pa'])
    
    # rff = np.linspace(3*G, 7*G, 101)
    # run.Variable('JPM A Fast Pulse Amplitude', 0.31)
    # rfa = np.linspace(0, 1, 51);
    # run.Sweep('RF Amplitude', rfa,
                # Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
                
    # run.Variable('JPM A Fast Pulse Amplitude', 0.315)
    # rfa = np.linspace(0, 1, 51);
    # run.Sweep('RF Amplitude', rfa,
                # Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
                
    # run.Variable('JPM A Fast Pulse Amplitude', 0.32)
    # rfa = np.linspace(0, 1, 51);
    # run.Sweep('RF Amplitude', rfa,
                # Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
                
                
    # run.Variable('JPM A Fast Pulse Amplitude', 0.325)
    # rfa = np.linspace(0, 1, 51);
    # run.Sweep('RF Amplitude', rfa,
                # Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])
                
    # run.Variable('JPM A Fast Pulse Amplitude', 0.33)
    # rfa = np.linspace(0, 1, 51);
    # run.Sweep('RF Amplitude', rfa,
                # Save=True, PrintData='Pa', PlotData='Pa', Dependencies=['Pa', 'JPM A Detection Time'])

    # run.Variable('JPM A Fast Pulse Amplitude', 0.313)
    # rff = np.linspace(3*G, 6*G, 151)
    # fpa = np.linspace(.3, .33, 31)
    # run.Sweep(['JPM A Fast Pulse Amplitude', 'RF Frequency'], [fpa, rff], 
                 # Save=True, PrintData='Pa',
                 # Dependencies=['Pa', 'JPM A Detection Time'])
                  
    # rff = np.linspace(.5*G, 5.5*G, 201)
    # fpa_A = np.linspace(.054, .084, 51)
    # fpa_B = np.linspace(.070, .090, 51)
    # run.Sweep([['JPM A Fast Pulse Amplitude', 'RF Frequency'], 
                 # ['JPM B Fast Pulse Amplitude', 'RF Frequency']],
                 # [[fpa_A, rff], [fpa_B, rff]], 
                 # Save=True, PrintData=['Pa', 'Pb'],
                 # Dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])
    
    # fpt = np.linspace(0, 1000, 251)
    # fpa_A = np.linspace(.052, .077, 51)
    # fpa_B = np.linspace(.072, .088, 51)
    # run.Sweep([['JPM A Fast Pulse Amplitude', 'JPM A Fast Pulse Time'], 
                 # ['JPM B Fast Pulse Amplitude', 'JPM B Fast Pulse Time']],
                 # [[fpa_A, fpt], [fpa_B, fpt]],
                 # Save=True, PrintData=['Pa', 'Pb'],
                 # Dependencies=[['Pa', 'JPM A Detection Time'], ['Pb', 'JPM B Detection Time']])