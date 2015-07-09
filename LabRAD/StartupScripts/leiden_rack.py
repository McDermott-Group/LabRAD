import os
import sys
import subprocess as sp
from msvcrt import getch, kbhit

import labrad as lr

LABRAD_HOST = 'localhost'
LABRAD_PORT = 7682
LABRAD_PASSWORD = ''
LABRAD_NODE = 'Leiden' 

# Absolute path to the LabVIEW program including its name.
LABVIEW_PATH_FILENAME = r'C:\Program Files (x86)\National Instruments\LabVIEW 2013\LabVIEW.exe'

# Absolute path to the folder with LabRAD.
LABRAD_PATH = r'C:\Users\5107-1\Desktop\Git Repositories\LabRAD\LabRAD'

# Relative paths with respect to LABRAD_PATH.
DIRECT_ETHERNET_SERVER_PATH = r'Servers\DirectEthernet'
TWISTD_PATH = r'C:\Python27\Lib\site-packages\twisted\scripts'
LABRAD_NODE_PATH = r'StartupScripts'
LABRAD_NODE_SERVERS_PATH = r'StartupScripts'
GHZ_FPGA_BRING_UP_PATH = r'Servers\Instruments\GHzBoards'
DC_RACK_LABVIEW_VI_PATH = r'Servers\Instruments\DCRack'

# Corresponding file names with the extensions.
LABRAD_FILENAME = 'LabRAD-v1.1.4.exe'
TWISTD_FILENAME = 'twistd.py'
LABRAD_NODE_FILENAME = 'labradnode.py'
LABRAD_NODE_SERVERS_FILENAME = 'labradnode_servers.py'
DIRECT_ETHERNET_SERVER_FILENAME = 'DirectEthernet.exe'
GHZ_FPGA_BRING_UP_FILENAME = 'ghz_fpga_bringup.py'
DC_RACK_LABVIEW_VI_FILENAME = 'DC_Rack_Control.vi'

class QuitException(Exception): pass

class LabRADServers:
    def __init__(self):
        if not os.path.exists(LABRAD_PATH):
            raise Exception('LABRAD_PATH = ' + LABRAD_PATH + ' does not exist.')
        self.processes = {}
 
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is not None:
            print('Closing the started processes...')
            for process in self.processes:
                return_code = self.processes[process].poll()
                # print(process + ": " + str(return_code))
                if return_code is None:
                    self.processes[process].terminate()
        if exception_type == QuitException:
            return True
            
    def _waitTillEnterKeyIsPressed(self):
        while kbhit():
            getch()
        print('\n\t[ENTER]: Continue.\n\t[Q]:\t Quit and close the started processes.\n')
        cont = True
        while cont: 
            while not kbhit():
                pass
            ch = getch()
            if ord(ch) == 13:
                cont = False
            if ord(ch) == 81 or ord(ch) == 113:
                raise QuitException('The user chose to quit.')
    
    def startLabRAD(self):
        print('Starting LabRAD...')
        labrad_filename = os.path.join(LABRAD_PATH, LABRAD_FILENAME)
        
        if not os.path.isfile(labrad_filename):
            raise Exception('Cannot locate the LabRAD sys.executable file ' + labrad_filename + '.')
        
        try:
            self.processes['LabRAD'] = sp.Popen(labrad_filename)
        except OSError:
            print('Failed to start LabRAD.')
            raise
        
        print('Please press "Run Server" button in the LabRAD window if it is not started automatically.')
        self._waitTillEnterKeyIsPressed()
        
    def startLabRADNode(self):
        print('Starting the LabRAD node...')
        node_filename = os.path.join(LABRAD_PATH, LABRAD_NODE_PATH, LABRAD_NODE_FILENAME)
        
        try:
            self.processes['LabRAD Node'] = sp.Popen([sys.executable, node_filename], creationflags=sp.CREATE_NEW_CONSOLE)
        except OSError:
            print('Failed to start the LabRAD node.')
            raise
        
        print('Please enter the password in the LabRAD node window that poped up.')
        print('Do not close the window when you are done.')
        self._waitTillEnterKeyIsPressed()
        
    def startLabRADNodeServers(self):
        print('Starting the servers with the Labrad node...')
        node_servers_filename = os.path.join(LABRAD_PATH, LABRAD_NODE_SERVERS_PATH, LABRAD_NODE_SERVERS_FILENAME)
        
        try:
            self.processes['LabRAD Node Servers'] = sp.Popen([sys.executable, node_servers_filename], creationflags=sp.CREATE_NEW_CONSOLE)
        except OSError:
            print('Failed to start the LabRAD node.')
            raise
        
        print('Please enter the password in the window that poped up.')
        print('The window will close automatically when the servers are started.')
        while self.processes['LabRAD Node Servers'].poll() is None:
            pass
        print('The servers have been started.\n')
        
    def startDirectEthernetServer(self):
        print('Starting Direct Ethernet server...')
        direct_ethernet = os.path.join(LABRAD_PATH, DIRECT_ETHERNET_SERVER_PATH, DIRECT_ETHERNET_SERVER_FILENAME)
        
        if not os.path.isfile(direct_ethernet):
            raise Exception('Cannot locate the Direct Ethernet Server sys.executable file ' + direct_ethernet + '.')
        
        try:
            self.processes['Direct Ethernet Server'] = sp.Popen(direct_ethernet)
        except OSError:
            print('Failed to start Direct Ethernet Server.')
            raise
        
        print('In Direct Ethernet window please specify the following:' +
              '\n\tLabRAD host name, e.g. "' + LABRAD_HOST + 
              '",\n\tLabRAD port, e.g. "' + str(LABRAD_PORT) + 
              '",\n\tyour password, e.g. "' + LABRAD_PASSWORD +
              '",\n\tand the LabRAD node name, e.g. "' + LABRAD_NODE + '".')
        self._waitTillEnterKeyIsPressed()
        
    def bringUpGHzFPGAs(self):
        print('Starting the GHz FPGA bring-up script...')
        bring_up = os.path.join(LABRAD_PATH, GHZ_FPGA_BRING_UP_PATH, GHZ_FPGA_BRING_UP_FILENAME)
         
        if not os.path.isfile(bring_up):
            raise Exception('Cannot locate the GHz FPGA bring-up script ' + bring_up + '.')
        
        try:
            self.processes['GHz FPGA Bring Up'] = sp.Popen([sys.executable, bring_up], creationflags=sp.CREATE_NEW_CONSOLE)
        except OSError:
            print('Failed to start the GHz FPGA bring up script.')
            raise
        
        print('Please enter the password in the GHz FPGA bring-up window and follow the instructions there.')
        print('Close the GHz FPGA bring-up window when you are done.\n')
        while self.processes['GHz FPGA Bring Up'].poll() is None:
            pass

    def startDCRackLabVIEWVI(self):
        print('Starting the DC Rack LabVIEW VI...')
        dc_rack_vi = os.path.join(LABRAD_PATH, DC_RACK_LABVIEW_VI_PATH, DC_RACK_LABVIEW_VI_FILENAME)
         
        if not os.path.isfile(dc_rack_vi):
            raise Exception('Cannot locate the DC Rack LabVIEW VI ' + dc_rack_vi  + '.')
        
        try:
            self.processes['DC Rack LabVIEW VI'] = sp.Popen('"' + LABVIEW_PATH_FILENAME + '" "' + dc_rack_vi + '"')
        except OSError:
            print('Failed to start the DC Rack LabVIEW VI.')
            raise
        
        print('Please press "Run" button in the LabVIEW VI window.')
        self._waitTillEnterKeyIsPressed()

# Define a main() function
def main():
    with LabRADServers() as inst:
        inst.startLabRAD()
        inst.startDirectEthernetServer()
        inst.startLabRADNode()
        inst.startLabRADNodeServers()
        inst.bringUpGHzFPGAs()
        inst.startDCRackLabVIEWVI()
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()