# JPM read out of a qubit connected to a resonator.

import sys
import numpy as np

import labrad as lr

import JPMTwoPhotonSpecExpt as two_photon

G = float(10**9)
M = float(10**6)

# Connect to LabRAD.
cxn = lr.connect()

# List of the experiment resources. Simply uncomment/comment the devices that should be used/unused.
# However, 'Resource': 'LabRAD Server' should never be left out.
Resources = [   { # This is a required resource.
                    'Resource': 'LabRAD Server', 
                    'Server Name': 'ghz_fpgas'
                },
                { # List here all parameters that specify waveforms.
                    'Resource': 'Waveform Parameters',
                    'Variables': [
                                    'Init Time', 'Bias Time', 'Measure Time',
                                    'Bias Voltage', 'Fast Pulse Time', 'Fast Pulse Amplitude', 'Fast Pulse Width',
                                    'RF1 SB Frequency', 'RF1 Amplitude', 'RF1 Time',
                                    'RF2 SB Frequency', 'RF2 Amplitude', 'RF2 Time',
                                    'RF1 to RF2 Delay', 'RF2 to Fast Pulse Delay'
                                 ]
                },
                { # DACs are converted to a simple ordered list internally based on 'List Index' value.
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 3',
                    'List Index': 0,
                    'DAC Settings': {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'RF1 I',
                                        'FO1 FastBias Firmware Version': '2.1'
                                    },
                    'Variables': []
                },
                {
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 4',
                    'List Index': 1,
                    'DAC Settings': {   
                                        'DAC A': 'RF2 Q',
                                        'DAC B': 'RF2 I'
                                    },
                    'Variables': []
                },
                { # GPIB RF Generator
                    'Resource': 'RF Generator',
                    'GPIB Address': 'GPIB0::20',
                    'Variables': ['RF1 Power', 'RF1 Frequency']
                },
                { # GPIB RF Generator
                    'Resource': 'RF Generator',
                    'GPIB Address': 'GPIB0::19',
                    'Variables': ['RF2 Power', 'RF2 Frequency']
                },
                { # Lab Brick Attenuator
                    'Resource': 'Lab Brick',
                    'Serial Number': 7033,
                    'Variables': ['RF1 Attenuation']
                },
                { # Lab Brick Attenuator
                    'Resource': 'Lab Brick',
                    'Serial Number': 7032,
                    'Variables': ['RF2 Attenuation']
                },
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
            'Experiment Name': 'FPAFPT2D',
            'Comments': 'RF1 = JPM RF Bias, RF2 = Qubit Readout' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 3000, # should not exceed ~50,000
            
            'RF1 Frequency': 20*G, #3.72*G, # Hz
            'RF1 Power': -110, # dBm
            'RF1 Attenuation': 63, # dB, should be in (0, 63] range
            'RF1 SB Frequency': 0*M, # Hz
            'RF1 Amplitude': 0, # DAC units
            'RF1 Time': 600, # ns
            
            'RF1 to RF2 Delay': 100, # ns

            'RF2 Frequency': 20*G, # Hz
            'RF2 Power': -110, # dBm
            'RF2 Attenuation': 63, # dB, should be in (0, 63] range
            'RF2 SB Frequency': 30*M, # Hz
            'RF2 Amplitude': 0, # DAC units
            'RF2 Time': 1000, # ns
          
            'RF2 to Fast Pulse Delay': 500, # ns (time delay between the start of the RF2 pulse and the start of the fast pulse).
          
            'Init Time': 500, # microseconds
            'Bias Time': 100, # microseconds
            'Measure Time': 50, # microseconds
          
            'Bias Voltage': 0.198, # FastBias DAC units
            'Fast Pulse Time': 10, # nanoseconds
            'Fast Pulse Amplitude': 0.42, # DAC units
            'Fast Pulse Width': 0, #nanoseconds
          
            'Threshold': 500, # Preamp Time Counts
            'Temperature': 11.5 # mK
           }

with two_photon.JPMTwoPhotonExpt(cxn) as run:
    
    run.SetExperiment(ExptInfo, Resources, ExptVars)         

    # run.Sweep('Bias Voltage', np.linspace(0.15, 0.20, 201), 
                # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])

    # run.Sweep('Fast Pulse Amplitude', np.linspace(0, 1.0, 101),
               # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
  
    # run.Sweep('Fast Pulse Time', np.linspace(0, 500, 501),
               # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
    
    run.Sweep(['Fast Pulse Amplitude', 'Fast Pulse Time'],
          [np.linspace(0.36, 0.52, 33), np.linspace(0, 1000, 101)], 
          Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep('RF2 Frequency', np.linspace(4.92*G, 5.06*G, 141), 
               # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
    
    # run.Sweep('RF2 Attenuation', np.linspace(7.4, 7.4, 1), 
            # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
            
    # run.Sweep(['RF1 Attenuation', 'Bias Voltage'],
           # [np.linspace(1, 63, 63), np.linspace(.16, .21, 101)], 
           # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['RF1 Attenuation', 'RF1 Frequency'],
                # [np.linspace(1, 63, 32), np.linspace(3*G, 7*G, 201)], 
                # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF1 Power', -110)
    # run.Variable('RF1 Frequency', 20*G)
    # run.Variable('RF1 Attenuation', 63)
    # run.Variable('RF1 Amplitude', 0)
    
    # run.Variable('RF2 Power', 13)
    # run.Variable('RF2 Amplitude', 0.5)
    
    # run.Sweep(['RF2 Attenuation', 'RF2 Frequency'],
                # [np.linspace(1, 63, 32), np.linspace(3*G, 7*G, 201)], 
                # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Frequency'],
               # [np.linspace(0, 1, 21), np.linspace(3*G, 5*G, 201)], 
               # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0)
    
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Frequency'],
           # [np.linspace(0, 1, 21), np.linspace(3*G, 5*G, 201)], 
           # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Attenuation'],
              # [np.linspace(0.3, 0.6, 61), np.linspace(1, 15, 15)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0.66)
    
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Attenuation'],
              # [np.linspace(0.3, 0.6, 61), np.linspace(1, 15, 15)], 
              # Save=True, PrintData=['Switching Probability'])
              
    # run.Variable('RF1 Amplitude', 0.5)
    # run.Variable('RF2 Amplitude', 0.0)

    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Attenuation'],
              # [np.linspace(0.3, 0.6, 31), np.linspace(1, 15, 15)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0.66)
    
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Attenuation'],
              # [np.linspace(0.3, 0.6, 31), np.linspace(1, 15, 15)], 
              # Save=True, PrintData=['Switching Probability'])
              
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Frequency'],
              # [np.linspace(0.36, 0.52, 33), np.linspace(3*G, 4.5*G, 151)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0.66)
    
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Frequency'],
              # [np.linspace(0.36, 0.52, 33), np.linspace(3*G, 4.5*G, 151)], 
              # Save=True, PrintData=['Switching Probability'])
              
    # run.Variable('RF1 Amplitude', 0.5)
    # run.Variable('RF2 Amplitude', 0.0)

    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Frequency'],
              # [np.linspace(0.36, 0.52, 33), np.linspace(3*G, 4.5*G, 151)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0.66)
    
    # run.Sweep(['Fast Pulse Amplitude', 'RF1 Frequency'],
              # [np.linspace(0.36, 0.52, 33), np.linspace(3*G, 4.5*G, 151)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep(['RF1 Amplitude', 'RF1 to RF2 Delay'],
              # [np.linspace(-0.1, 1.0, 45), np.linspace(-510, 100, 62)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0.66)
    
    # run.Sweep(['RF1 Amplitude', 'RF1 to RF2 Delay'],
              # [np.linspace(-0.1, 1.0, 45), np.linspace(-510, 100, 62)], 
              # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep(['RF1 Frequency', 'RF2 Frequency'],
               # [np.linspace(3*G, 7*G, 101), np.linspace(3*G, 7*G, 101)], 
               # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('Fast Pulse Amplitude', 0.4)
    
    # run.Sweep(['RF1 Frequency', 'RF2 Frequency'],
           # [np.linspace(3*G, 7*G, 101), np.linspace(3*G, 7*G, 101)], 
           # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('Fast Pulse Amplitude', 0.5)
    
    # run.Sweep(['RF1 Frequency', 'RF2 Frequency'],
           # [np.linspace(3*G, 7*G, 101), np.linspace(3*G, 7*G, 101)], 
           # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['RF1 Flux Bias Voltage', 'RF2 Frequency'],
               # [np.linspace(0, 1, 21), np.linspace(4.98*G, 5*G, 101)], 
               # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep('RF1 Time', np.linspace(0, 1000, 501),
              # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
          
    #run.Sweep('RF1 Drive to RF2 Delay', np.linspace(0, 8000, 401),
    #      Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
            
    # run.Sweep(['RF1 Attenuation', 'RF1 Time'], [np.linspace(1, 11, 11), np.linspace(0, 400, 401)],
        # Save=True, PrintData=['Switching Probability'])
        
    # run.Sweep(['RF2 Frequency', 'RF2 Amplitude'], 
               # [np.linspace(4.98*G, 5*G, 81), np.linspace(0, 0.5, 51)],
               # Save=True, PrintData=['Switching Probability'])
               
    #run.Sweep(['Fast Pulse Amplitude', 'RF2 Amplitude'], 
    #           [np.linspace(0.26, 0.32, 61), np.linspace(0, 0.5, 51)],
    #           Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Amplitude', 0.5)
    
    # run.Sweep('Fast Pulse Amplitude', np.linspace(0.30, 0.37, 141),
           # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Power', -110)
    # run.Variable('RF2 Attenuation', 63)
    # run.Variable('RF2 Amplitude', 0)
               
    # run.Sweep('Fast Pulse Amplitude', np.linspace(0.30, 0.37, 141),
           # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['Fast Pulse Amplitude', 'Fast Pulse Time'], 
           # [np.linspace(0.28, 0.36, 81), np.linspace(0, 20, 21)],
           # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF2 Power', -110)
    # run.Variable('RF2 Attenuation', 63)
    # run.Variable('RF2 Amplitude', 0)
               
    # run.Sweep(['Fast Pulse Amplitude', 'Fast Pulse Time'], 
               # [np.linspace(0.28, 0.36, 81), np.linspace(0, 20, 21)],
               # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep(['RF2 Frequency', 'RF2 Time'],
               # [4.991*G + np.linspace(-3*M, 3*M, 21), np.linspace(0, 800, 81)],
               # Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('RF1 Time', 96)
    # run.Variable('RF1 Amplitude', 0.5)
    
    # run.Sweep(['RF2 Frequency', 'RF2 Time'],
               # [4.991*G + np.linspace(-3*M, 3*M, 21), np.linspace(0, 800, 81)],
               # Save=True, PrintData=['Switching Probability'])
               
    # run.Sweep(['Displacement Phase', 'Displacement Time'],
           # [np.linspace(0, 2 * np.pi, 40), np.linspace(0, 50, 51)],
           # Save=True, PrintData=['Switching Probability'])