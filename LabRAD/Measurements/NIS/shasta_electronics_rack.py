# Read out of an NIS junction connected to a resonator.

import os
import numpy as np

from labrad.units import us, ns, mV, V, GHz, MHz, rad, dB, dBm, DACUnits

import nis_electronics_rack


comp_name = os.environ['COMPUTERNAME'].lower()
Resources = [ {
                'Interface': 'GHz FPGA Boards',
                'Boards': [
                            'mcdermott5125 Board DAC 9', 
                            'mcdermott5125 Board DAC 10',
                            'mcdermott5125 Board ADC 11'
                          ],
                'mcdermott5125 Board DAC 9': {
                                        'DAC A': 'RF I',    # DAC A
                                        'DAC B': 'RF Q',    # DAC B
                                        'FO1 FastBias Firmware Version': '2.1',
                                        'FO2 FastBias Firmware Version': '2.1',
                                      },
                'mcdermott5125 Board DAC 10': {
                                        'DAC A': 'None',
                                        'DAC B': 'None',
                                      },
                'mcdermott5125 Board ADC 11': {
                                        'RunMode': 'demodulate', #'average'
                                        'FilterType': 'square',
                                        'FilterLength': 6500 * ns,
                                        'DemodPhase': 0 * rad,
                                        'DemodCosAmp': 255,
                                        'DemodSinAmp': 255,
                                        'DemodFreq': -31.5 * MHz,
                                        'ADCDelay': 0 * ns,
                                        'Data': True
                                      },
                'Variables': {  # These are default values, you should
                                # normally overwrite these values in
                                # your particular experiment run.
                                'Init Time': {'Value': 100 * us},
                                'NIS Bias Voltage': {'Value': 0 * V},
                                'NIS Bias Time': {'Value': 10 * us},
                                'RF Amplitude': {'Value': 0 * DACUnits},
                                'RF Time': {'Value': 0 * ns},
                                'RF SB Frequency': {'Value': 31.5 * MHz},
                                'Bias to RF Delay': {'Value': 0 * us},
                                'ADC Wait Time': {'Value': 0 * ns},
                                'ADC Filter Length': {'Value': 10000 * ns},
                                'Calibration Coefficient': {'Value': 0}
                             }
                },
                { # GPIB RF Generator, 'Address' field is required only
                  # when more than one GPIB RF generator is present.
                    'Interface': 'RF Generator',
                    'Variables': {  
                                    'RF Power': {'Setting': 'Power'}, 
                                    'RF Frequency': {'Setting': 'Frequency'}
                                 }
                },
                # { # Lab Brick Attenuator
                    # 'Interface': 'Lab Brick Attenuator',
                    # 'Serial Number': 7031,
                    # 'Variables': 'Qubit Attenuation'
                # },
                { # ADR3
                    'Interface': 'ADR3',
                    'Variables': {'Temperature': {'Setting':'Temperatures', 'Stage': 'FAA'}}
                },
                { # Readings entered manually, software parameters.
                    'Interface': None,
                    'Variables': ['Reps', 'Runs'],
                }
            ]

# Experiment Information
ExptInfo = {
            'Device Name': 'NISDirectInjection',
            'User': 'Umesh',
            'Base Path': 'Z:\mcdermott-group\Data\NIS Junctions\NISDirectInjection',
            'Experiment Name': 'PowerSweep1nW',
           'Comments': 'PowerSweep1nW' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 480, # should not exceed ~5,000, use argument "runs" in sweep parameters instead 

            'Init Time': 3000 * us,

            'RF Frequency': 5.1683 * GHz,
            'RF Power': 16.5 * dBm, #17.6 * dBm,
            'RF Time': 10000 * ns, #17000 * ns,
            'RF SB Frequency': 0 * MHz, # 31.5 * MHz,
            'RF Amplitude': 1 * DACUnits, #0.5 * DACUnits, # [-1, 1] * DACUnits, 1 DACUnits ~ 0.1-2.0 V
            
            'NIS Bias Voltage': 0.2 * V, # -2.5 to 2.5 V or 0 to 5 V
            'NIS Bias Time': 50 * us, #600 * us,
            
            'Bias to RF Delay': 10 * us, #400 * us,
     
            'ADC Wait Time': 4000 * ns,
            'ADC Filter Length': 2000 * ns
           }


# with nis_electronics_rack.NISReadout() as run:
with nis_electronics_rack.NISReadoutRelaxation() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    # Direct Injection
    
    # run.sweep('RF Frequency', 
                 # np.linspace(5.1683,5.1689, 80) * GHz, print_data=['I', 'Q','Amplitude'],
                 # plot_data=['I', 'Q', 'Amplitude'],
                 # save=True, runs=1) 
                 
    # run.sweep('RF Frequency', 
                 # np.linspace(4.8735, 4.8765, 500) * GHz, print_data=['I', 'Q','Amplitude'],
                 # plot_data=['I', 'Q', 'Amplitude'],
                 # save=True, runs=1)             
                 
    # bias_range = np.linspace(0.1,0.5,3) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)
        # run.sweep(['RF Amplitude', 'RF Frequency'], 
                  # [np.linspace(0.1,0.5,5) * DACUnits, np.linspace(4.875, 4.8775, 300) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)  

    
    # run.value('NIS Bias Time', 300 * us)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(-30,50,9) * us, np.linspace(5.422,5.4250, 500) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
                  
    # run.value('NIS Bias Time', 300 * us)
    run.sweep(['Bias to RF Delay'], 
                  [np.linspace(0,350,36) * us], print_data=['I', 'Q','Amplitude'],
                  save=True, runs=1)   

    # run.value('NIS Bias Time', 750 * us)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(180,600,15) * us, np.linspace(4.8735, 4.8765, 500) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    
    
    # run.value('NIS Bias Voltage', 0.0 * V)
    # run.sweep('RF Frequency', 
                 # np.linspace(4.8735, 4.8765, 500) * GHz, print_data=['I', 'Q','Amplitude'],
                 # plot_data=['I', 'Q', 'Amplitude'],
                 # save=True, runs=1)
                  
    # run.value('NIS Bias Time', 670 * us)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(700,1200,6) * us, np.linspace(4.8795, 4.8840, 420) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(40,100,3) * us, np.linspace(4.879, 4.8840, 380) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)              
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(200,400,3) * us, np.linspace(4.879, 4.8840, 380) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1) 
                  
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(150,500,8) * us, np.linspace(4.879, 4.8840, 380) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)              
                  
    # run.value('NIS Bias Voltage', 0.0375 * V)
    # run.value('NIS Bias Time', 850 * us)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,200,11) * us, np.linspace(4.879, 4.8840, 450) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(250,800,12) * us, np.linspace(4.879, 4.8840, 400) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)   

    # run.value('NIS Bias Voltage', 0.0175 * V)
    # run.value('NIS Bias Time', 850 * us)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,200,11) * us, np.linspace(4.879, 4.8840, 450) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(250,800,12) * us, np.linspace(4.879, 4.8840, 400) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)              
                 
    # Direct Injection  
    
    
    # run.single_shot_iqs(save=False, plot_data=True)
    # run.single_shot_osc(save=False, plot_data=['I', 'Q'])
    # run.avg_osc(save=True, plot_data=['I', 'Q'], runs=1000)

    # run.sweep('ADC Wait Time', np.linspace(0, 10000, 251) * ns,
          # print_data=['I', 'Q', 'Temperature'], plot_data=['I', 'Q', 'Amplitude'],
          # max_data_dim=1, save=True, runs=1)
    
    # Increase RF Time to ~18 000 ns.
    # See how the SNR imporoves with longer ADC Filter Length
    # run.sweep('ADC Filter Length', np.linspace(200, 10200, 251) * ns,
          # print_data=['I', 'Q', 'Temperature'], plot_data=['I', 'Q', 'Amplitude'],
          # max_data_dim=1, save=True, runs=1)
    
    # run.sweep('RF Amplitude', np.linspace(0, 1, 31) * DACUnits,
              # print_data=['I', 'Q'], plot_data=['I', 'Q'], max_data_dim=1,
              # save=False, runs=1)
    
    # run.sweep('RF Frequency', np.linspace(4.8549, 4.8569,350) * GHz,
               # plot_data=['I', 'Q', 'Amplitude'], max_data_dim=1,
               # save=True, runs=1)
    
    # run.sweep(['Bias to Readout Delay', 'RF Frequency'],s
              # [np.linspace(0, 100, 101) * us, np.linspace(4.9, 5, 101) * GHz],
               # print_data=['I', 'Q'], plot_data=['I', 'Q'], 
               # save=True, runs=3) # runs does ex: 3X 4000 reps
               
   
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
         # [np.linspace(0, 100, 11) * us, np.linspace(4.6725, 4.6744, 200) * GHz],
         # save=True, runs=1)
         
    # bias_range = np.linspace(0.1, 0.5, 5) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)
        
    # run.sweep('RF Frequency', 
                 # np.linspace(4.854, 4.8568, 500) * GHz, print_data=['I', 'Q','Amplitude'],
                 # save=True, runs=1)    
        
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,200,11) * us, np.linspace(4.8543, 4.8568, 500) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(250,600,8) * us, np.linspace(4.8536, 4.8569, 430) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1) 
                  
    # run.value('NIS Bias Voltage', 0.08 * V)              
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,200,11) * us, np.linspace(4.8543, 4.8568, 500) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(250,600,8) * us, np.linspace(4.8536, 4.8569, 430) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)              
    
    # run.value('NIS Bias Voltage', 0.054 * V)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,300,31) * us, np.linspace(4.8543, 4.8568, 500) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(70,180,12) * us, np.linspace(4.8536, 4.8569, 450) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    
    
    
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(50,200,16) * us, np.linspace(4.8536, 4.8569, 380) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)             
                  
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(100,200,2) * us, np.linspace(4.8546, 4.8569, 470) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)              
        
    # run.sweep('NIS Bias Voltage', np.array([0.15]) * V, save=True)    
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,50,11) * us, np.linspace(4.854, 4.8569, 400) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(50,200,16) * us, np.linspace(4.854, 4.8569, 400) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)              
                  
    # run.sweep('NIS Bias Voltage', np.array([0]) * V, save=True)              
                  
    # bias_range = np.linspace(0.15, 0.3, 2) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)              
                  
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0,120,25) * us, np.linspace(4.854, 4.8569, 250) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
    
    # bias_range = np.linspace(0.1, 0.5, 5) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)
        # run.sweep(['RF Amplitude', 'RF Frequency'], 
                  # [np.linspace(0.1,0.5,5) * DACUnits, np.linspace(4.8549,4.8569,350) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)  
                  
    # run.sweep(['RF Amplitude', 'RF Frequency'], 
                  # [np.linspace(0.1,0.5,5) * DACUnits, np.linspace(4.8549,4.8569,350) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)               
                  
                  
    # run.sweep(['RF Amplitude', 'RF Frequency'], 
         # [np.linspace(0.2,0.8,4) * DACUnits, np.linspace(4.8552, 4.8569,100) * GHz],
         # plot_data=['I', 'Q', 'Amplitude']
         # save=True, runs=1)     
         
    # run.sweep(['ADC Filter Length', 'RF Frequency'], 
         # [np.linspace(1000,6000,6) * ns, np.linspace(4.6725, 4.6744, 200) * GHz],
         # save=True, runs=1)         
            
    # bias_range = np.linspace(0.25, 0.5, 2) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)
        # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  # [np.linspace(0, 50, 51) * us, np.linspace(4.670, 4.675, 200) * GHz], print_data=['I', 'Q','Amplitude'],
                  # save=True, runs=1)
                  
    # bias_range = np.linspace(0, 1, 11) * V
    # for voltage in bias_range:
        # run.value('NIS Bias Voltage', voltage)
        # run.sweep('RF Frequency', np.linspace(4.673, 4.6744, 101) * GHz,
          # plot_data=['I', 'Q', 'Amplitude'], max_data_dim=1,
          # save=True, runs=1)