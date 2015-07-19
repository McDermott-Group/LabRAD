# JPM read out of a qubit connected to a resonator.

import numpy as np

from labrad.units import us, ns, V, GHz, MHz, dB, dBm, DACUnits, FastBiasUnits, PreAmpTimeCounts

import JPMQubitReadoutWithResetExpt as qr

# List of the experiment resources. Simply uncomment/comment the devices that should be used/unused.
# However, 'Resource': 'LabRAD Server' should never be left out.
Resources = [   { # Waveform parameters.
                    'Resource': 'GHz Boards', 
                    'Server': 'GHz FPGAs'
                    'Variables': [
                                    'Init Time', 'Bias Time', 'Measure Time',
                                    'Bias Voltage', 'Fast Pulse Time', 'Fast Pulse Amplitude', 'Fast Pulse Width',
                                    'Qubit SB Frequency', 'Qubit Amplitude', 'Qubit Time',
                                    'Readout SB Frequency', 'Readout Amplitude', 'Readout Time', 'Readout Phase',
                                    'Displacement Amplitude', 'Displacement Time', 'Displacement Phase',
                                    'Qubit Drive to Readout',  'Readout to Displacement', 'Displacement to Fast Pulse',
                                    'Readout to Displacement Offset'
                                 ]
                },
                { # DACs are converted to a simple ordered list internally based on 'List Index' value.
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 3',
                    'List Index': 0,
                    'DAC Settings': {
                                        'DAC A': 'JPM Fast Pulse',
                                        'DAC B': 'Qubit I',
                                        'FO1 FastBias Firmware Version': '2.1'
                                    },
                    'Variables': []
                },
                {
                    'Resource': 'DAC',
                    'DAC Name': 'Leiden Board DAC 4',
                    'List Index': 1,
                    'DAC Settings': {   
                                        'DAC A': 'Readout Q',
                                        'DAC B': 'Readout I'
                                    },
                    'Variables': []
                },
                # { # GPIB RF Generator.
                    # 'Resource': 'RF Generator',
                    # 'Server': 'GPIB RF Generators',
                    # 'Address': os.environ['COMPUTERNAME'] + ' GPIB Bus - GPIB0::19::INSTR',
                    # 'Variables': {'Readout Power': 'Power', 
                    #               'Readout Frequency': 'Frequency'}
                # },
                { # GPIB RF Generator.
                    'Resource': 'RF Generator',
                    'Server': 'GPIB RF Generators',
                    'Address': os.environ['COMPUTERNAME'] + ' GPIB Bus - GPIB0::20::INSTR',
                    'Variables': {'Qubit Power': 'Power', 
                                  'Qubit Frequency': 'Frequency'}
                },
                # { # GPIB RF Generator.
                    # 'Resource': 'RF Generator',
                    # 'Server': 'GPIB RF Generators',
                    # 'GPIB Address': 'GPIB0::20',
                    # 'Variables': {'RF Power': 'Power', 
                    #               'RF Frequency': 'Frequency'}
                # },
                # { # Lab Brick Attenuator.
                    # 'Resource': 'Lab Brick Attenuator',
                    # 'Server': os.environ['COMPUTERNAME'] + ' Lab Brick Attenuators',
                    # 'Serial Number': 7032,
                    # 'Variables': {'Readout Attenuation': 'Attenuation'}
                # },
                { # Lab Brick Attenuator.
                    'Resource': 'Lab Brick Attenuator',
                    'Server': os.environ['COMPUTERNAME'] + ' Lab Brick Attenuators',
                    'Address': 7032,
                    'Variables': 'Readout Attenuation'
                },
                { # Lab Brick Attenuator.
                    'Resource': 'Lab Brick Attenuator',
                    'Server': os.environ['COMPUTERNAME'] + ' Lab Brick Attenuators',
                    'Address': 7033,
                    'Variables': {'Qubit Attenuation': 'Attenuation'}
                },
                { # SIM Voltage Source.
                    'Resource': 'Voltage Source',
                    'Server': os.environ['COMPUTERNAME'] + ' SIM928',
                    'Address': 'GPIB0::26::SIM900::3',
                    'Variables': ['Qubit Flux Bias Voltage']
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
            'Device Name': 'MH036 - JPM 051215A-E10',
            'User': 'Ivan Pechenezhskiy',
            'Base Path': 'Z:\mcdermott-group\Data\Syracuse Qubits',
            'Experiment Name': 'RFreq1D',
            'Comments': 'RF generator - DAC - attenuators - 50 dB - LP filter - qubit - LP filter - isolator - JPM.' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 3000, # should not exceed ~50,000
          
            'Qubit Frequency': 20 * GHz,
            'Qubit Power': -110 * dBm, 
            'Qubit Attenuation': 63 * dB, # should be in (0, 63] range
            'Qubit SB Frequency': 0 * MHz,
            'Qubit Amplitude': 0.5 * DACUnits
            'Qubit Time': 8000 * ns
            
            'Qubit Drive to Readout': 0 * ns,
            
            'Qubit Flux Bias Voltage': 0 * V,

            'Readout Frequency': 20 * GHz,
            'Readout Power': 13 * dBm,
            'Readout Attenuation': 1 * dB, # should be in (0, 63] range
            'Readout SB Frequency': 0 * MHz, 
            'Readout Amplitude': 1 * DACUnits,
            'Readout Time': 1000 * ns
            'Readout Phase': 0 * rad
            
            'Readout to Displacement': 0 * ns,
            'Readout to Displacement Offset': 0.00 * DACUnits,
            
            'Displacement Amplitude': 0.0 * DACUnits,
            'Displacement Time': 0 * ns,
            'Displacement Phase': 0 * rad,
            
            'Displacement to Fast Pulse': -100 * ns,  # time delay between the end of the displacement pulse and the start of the fast pulse
          
            'Init Time': 500 * us,
            'Bias Time': 100 * us,
            'Measure Time': 50 * us,
          
            'Bias Voltage': 0.194 * V,
            'Fast Pulse Time': 10 * ns,
            'Fast Pulse Amplitude': 0.2205 * DACUnits,
            'Fast Pulse Width': 0 * ns,
          
            'Threshold': 100 * counts,
            'Temperature': 14.2 * mK
           }

with qr.JPMQubitReadoutWithReset() as run:
    
    run.SetExperiment(ExptInfo, Resources, ExptVars) 

    # run.Sweep('Bias Voltage', np.linspace(0.1, 0.3, 101), 
                # Save=False, PrintData=['Switching Probability'], PlotData=['Switching Probability'])

    # run.Sweep('Fast Pulse Amplitude', np.linspace(0.2, .25, 201),
            # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
  
    # run.Sweep('Fast Pulse Time', np.linspace(0, 500, 501),
               # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
  
    # run.Sweep('Displacement to Fast Pulse', np.linspace(-28, 72, 101),
            # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])

    run.Sweep('Readout Frequency', np.linspace(4.3, 5.3, 5001) * GHz, 
              Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])

    # run.Sweep('Readout to Displacement Offset', np.linspace(-0.1, .2, 61), 
            # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])

    # run.Sweep('Readout Attenuation', np.linspace(1, 21, 41), 
            # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
            
    # run.Sweep('Readout Time', np.linspace(0, 800, 401), 
              # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
    
    # run.Sweep(['Readout Attenuation', 'Readout Time'],
          # [np.linspace(1, 46, 91), np.linspace(1, 15, 15)], 
           # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['Readout Attenuation', 'Readout Frequency'],
              # [np.linspace(30, 60, 21), np.linspace(4.98*G, 5*G, 101)], 
               # Save=True, PrintData=['Switching Probability'])
               
    # run.Sweep('Displacement to Fast Pulse', np.linspace(-600, 600, 241), 
               # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
               
    # run.Sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'],
               # [np.linspace(0, 2, 21), np.linspace(4.98*G, 5*G, 101)],
               # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['RF Attenuation', 'RF Frequency'],
                # [np.linspace(3, 43, 41), np.linspace(3*G, 5*G, 201)], 
                # Save=True, PrintData=['Switching Probability'])

    # run.Variable('Readout Amplitude', 0.66)
    
    # run.Sweep(['RF Attenuation', 'RF Frequency'],
            # [np.linspace(3, 43, 41), np.linspace(3*G, 5*G, 201)], 
            # Save=True, PrintData=['Switching Probability'])
            
    # run.Sweep(['Bias Voltage', 'RF Attenuation'],
                # [np.linspace(0., 0.2, 81), np.linspace(0, 30, 31)], 
                # Save=True, PrintData=['Switching Probability'])

    # run.Variable('Readout Amplitude', 0.66)
    
    # run.Sweep(['Bias Voltage', 'RF Attenuation'],
            # [np.linspace(0., 0.2, 81), np.linspace(0, 30, 31)], 
            # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep(['RF Frequency', 'Fast Pulse Amplitude'],
              # [np.linspace(3*G, 5*G, 201), np.linspace(0, 0.5, 51)], 
                # Save=True, PrintData=['Switching Probability'])

    # run.Variable('Readout Amplitude', 0.66)
    
    # run.Sweep(['RF Frequency', 'Fast Pulse Amplitude'],
              # [np.linspace(3*G, 5*G, 201), np.linspace(0, 1, 51)], 
                # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['Readout Attenuation', 'Readout Frequency'],
            # [np.linspace(1, 19, 10), np.linspace(4.98*G, 5.0*G, 101)], 
            # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep(['Qubit Flux Bias Voltage', 'Qubit Frequency'],
               # [np.linspace(0, .4, 11), np.linspace(4*G, 4.9*G, 451)], 
               # Save=True, PrintData=['Switching Probability'])
               
    # run.Sweep(['Qubit Flux Bias Voltage', 'Readout Frequency'],
               # [np.linspace(0, 1, 21), np.linspace(4.98*G, 5*G, 101)], 
               # Save=True, PrintData=['Switching Probability'])
    
    # run.Sweep('Qubit Time', np.linspace(0, 1000, 501),
              # Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
          
    #run.Sweep('Qubit Drive to Readout Delay', np.linspace(0, 8000, 401),
    #      Save=True, PrintData=['Switching Probability'], PlotData=['Switching Probability'])
            
    # run.Sweep(['Qubit Attenuation', 'Qubit Time'], [np.linspace(1, 11, 11), np.linspace(0, 400, 401)],
        # Save=True, PrintData=['Switching Probability'])
        
    # run.Sweep(['Readout Frequency', 'Readout Amplitude'], 
               # [np.linspace(4.98*G, 5*G, 81), np.linspace(0, 0.5, 51)],
               # Save=True, PrintData=['Switching Probability'])
               
    #run.Sweep(['Fast Pulse Amplitude', 'Readout Amplitude'], 
    #           [np.linspace(0.26, 0.32, 61), np.linspace(0, 0.5, 51)],
    #           Save=True, PrintData=['Switching Probability'])
    
    # run.Variable('Readout Amplitude', 0.5)
    
    # run.Sweep('Fast Pulse Amplitude', np.linspace(0.30, 0.37, 141),
           # Save=True, PrintData=['Switching Probability'])

    # run.Sweep(['Displacement Phase', 'Displacement Amplitude'],
           # [np.linspace(0, 2 * np.pi, 50), np.linspace(0, 1, 51)],
           # Save=True, PrintData=['Switching Probability'])
        
    # run.Sweep(['Readout to Displacement Offset', 'Displacement Amplitude'],
           # [np.linspace(-.1, .2, 61), np.linspace(0, 1, 41)],
           # Save=True, PrintData=['Switching Probability'])
           
   