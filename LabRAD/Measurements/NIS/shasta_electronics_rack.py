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
                                'ADC Filter Length': {'Value': 10000 * ns}
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
            'Device Name': 'NISExtenTrap',
            'User': 'Umesh',
            'Base Path': 'Z:\mcdermott-group\Data\NIS Junctions',
            'Experiment Name': 'CodeTest',
            'Comments': '1D sweep Frequency vary Filter Width 10dB att removed at output' 
           }
 
# Experiment Variables
ExptVars = {
            'Reps': 150, # should not exceed ~5,000, use argument "runs" in sweep parameters instead 

            'Init Time': 3000 * us,

            'RF Frequency': 4.6736 * GHz,
            'RF Power': 16.5 * dBm, #17.6 * dBm,
            'RF Time': 17000 * ns,
            'RF SB Frequency': 31.5 * MHz,
            'RF Amplitude': 0.5 * DACUnits, # [-1, 1] * DACUnits, 1 DACUnits ~ 0.1-2.0 V
            
            'NIS Bias Voltage': 0.0 * V, # -2.5 to 2.5 V or 0 to 5 V
            'NIS Bias Time': 300 * us,
            
            'Bias to RF Delay': 100 * us,
     
            'ADC Wait Time': 4000 * ns,
            'ADC Filter Length': 2000 * ns
           }


with nis_electronics_rack.NISReadout() as run:
    
    run.set_experiment(ExptInfo, Resources, ExptVars)
    
    #run.single_shot_iqs(save=False, plot_data=True)
    #run.single_shot_osc(save=False, plot_data=['I', 'Q'])
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
    
    # run.sweep('RF Frequency', np.linspace(4.6722, 4.6745,350) * GHz,
               # plot_data=['I', 'Q', 'Amplitude'], max_data_dim=1,
               # save=True, runs=1)
    
    # run.sweep(['Bias to Readout Delay', 'RF Frequency'],s
              # [np.linspace(0, 100, 101) * us, np.linspace(4.9, 5, 101) * GHz],
               # print_data=['I', 'Q'], plot_data=['I', 'Q'], 
               # save=True, runs=3) # runs does ex: 3X 4000 reps
               
   
    # run.sweep(['Bias to RF Delay', 'RF Frequency'], 
         # [np.linspace(0, 100, 11) * us, np.linspace(4.6725, 4.6744, 200) * GHz],
         # save=True, runs=1)
         
    bias_range = np.linspace(0.25, 0.5, 2) * V
    for voltage in bias_range:
        run.value('NIS Bias Voltage', voltage)
        run.sweep(['Bias to RF Delay', 'RF Frequency'], 
                  [np.linspace(0, 50, 21) * us, np.linspace(4.6722, 4.6745, 350) * GHz], print_data=['I', 'Q','Amplitude'],
                  save=True, runs=1)     
         
            
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