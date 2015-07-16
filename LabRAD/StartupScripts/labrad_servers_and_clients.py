# Copyright (C) 2015 Ivan Pechenezhskiy
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

"""
This script can be used to start LabRAD manager, servers with the LabRAD node and some other programs. 
Run "labrad_servers_clients.py -h" in the command line for the command line input options.
"""

import os
import sys
import subprocess as sp
import argparse
from msvcrt import getch, kbhit

import labrad as lr

LABRAD_HOST = 'localhost'
LABRAD_PORT = 7682
LABRAD_PASSWORD = ''
LABRAD_NODE = 'Leiden'

# Absolute path to the folder with LabRAD.
LABRAD_PATH = os.path.join(os.environ['HOME'], r'Desktop\Git Repositories\LabRAD\LabRAD')

# Relative paths with respect to LABRAD_PATH.
DIRECT_ETHERNET_SERVER_PATH = r'Servers\DirectEthernet'
LABRAD_NODE_PATH = r'StartupScripts'
LABRAD_NODE_SERVERS_PATH = r'StartupScripts'
GHZ_FPGA_BRING_UP_PATH = r'Servers\Instruments\GHzBoards'
DC_RACK_LABVIEW_VI_PATH = r'Servers\Instruments\DCRack'

# Corresponding file names with the extensions.
LABRAD_FILENAME = 'LabRAD-v1.1.4.exe'
LABRAD_NODE_FILENAME = 'labradnode.py'
LABRAD_NODE_SERVERS_FILENAME = 'labradnode_servers.py'
DIRECT_ETHERNET_SERVER_FILENAME = 'DirectEthernet.exe'
GHZ_FPGA_BRING_UP_FILENAME = 'ghz_fpga_bringup.py'
DC_RACK_LABVIEW_VI_FILENAME = 'DC_Rack_Control.vi'

class QuitException(Exception): pass

class StartAndBringUp:
    def __init__(self):
        if not os.path.exists(LABRAD_PATH):
            raise Exception('LABRAD_PATH = ' + LABRAD_PATH + ' does not exist.')
        self.processes = {}
        self.args = self._parseArguments()
 
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
            
    def _parseArguments(self):
        parser = argparse.ArgumentParser(description='Start LabRAD, LabRAD servers and other programs.')
        parser.add_argument('--registry-path', 
                            nargs='*',
                            default=['Start Lists', os.environ['COMPUTERNAME'].lower()],
                            help='path in the LabRAD Registry to the key containing the list of programs to run;' +
                            " root folder name ''" + ' must be omitted (default: "Start Lists" "%%COMPUTERNAME%%")')
        parser.add_argument('--registry-start-list-key', 
                            default='Start Program List',
                            help='Registry key containing the list of programs to run (default: "Start Program List")')
        parser.add_argument('--registry-labview-path-key', 
                            default='LabVIEW Path',
                            help='Registry key with the LabVIEW Path (default: "LabVIEW Path")')
        return parser.parse_args()  
    
    def _LabRADConnect(self):
        print('Connecting to LabRAD...')
        try:
            return lr.connect()
        except:
            raise Exception('Cannot connect to LabRAD. The LabRAD program does not appear to be running.')
    
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
            if ord(ch) in [81, 113]:
                raise QuitException('The user chose to quit.')
    
    def startLabRAD(self):
        print('Starting LabRAD...')
        labrad_filename = os.path.join(LABRAD_PATH, LABRAD_FILENAME)
        
        if not os.path.isfile(labrad_filename):
            raise Exception('Cannot locate the LabRAD sys.executable file ' + labrad_filename + '.')
        
        try:
            self.processes['LabRAD'] = sp.Popen(labrad_filename)
        except OSError:
            raise Exception('Failed to start LabRAD.')
        
        print('Please press "Run Server" button in the LabRAD window if it is not started automatically.')
        self._waitTillEnterKeyIsPressed()
        
    def readProgramList(self):
        cxn = self._LabRADConnect()

        print('Getting the list of programs and servers to run from the LabRAD Registry...')
        try:
            cxn.registry.cd([''] + self.args.registry_path)
            return cxn.registry.get(self.args.registry_start_list_key)
        except:
            raise Exception('Could not read the LabRAD Registry. Please check that the Registry path ' + 
                  str([''] + self.args.registry_path) + ' and the key name ' + self.args.registry_start_list_key + ' are correct.')
        
    def startLabRADNode(self):
        print('Starting the LabRAD node...')
        node_filename = os.path.join(LABRAD_PATH, LABRAD_NODE_PATH, LABRAD_NODE_FILENAME)
        
        try:
            self.processes['LabRAD Node'] = sp.Popen([sys.executable, node_filename], creationflags=sp.CREATE_NEW_CONSOLE)
        except OSError:
            raise Exception('Failed to start the LabRAD node.')
        
        print('Please enter the password in the LabRAD node window that poped up.')
        print('Do not close the window when you are done.')
        self._waitTillEnterKeyIsPressed()
        
    def startLabRADNodeServers(self):
        print('Starting the servers with the Labrad node...')
        node_servers_filename = os.path.join(LABRAD_PATH, LABRAD_NODE_SERVERS_PATH, LABRAD_NODE_SERVERS_FILENAME)
        
        try:
            self.processes['LabRAD Node Servers'] = sp.Popen([sys.executable, node_servers_filename], creationflags=sp.CREATE_NEW_CONSOLE)
        except OSError:
            raise Exception('Failed to start the LabRAD node.')
        
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
            raise Exception('Failed to start Direct Ethernet Server.')
        
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
            raise Exception('Failed to start the GHz FPGA bring up script.')
        
        print('Please enter the password in the GHz FPGA bring-up window and follow the instructions there.')
        print('Close the GHz FPGA bring-up window when you are done.\n')
        while self.processes['GHz FPGA Bring Up'].poll() is None:
            pass

    def startDCRackLabVIEWVI(self):
        cxn = self._LabRADConnect()
        print('Getting the path to the LabVIEW.exe from the LabRAD Registry...')
        try:
            cxn.registry.cd([''] + self.args.registry_path)
            labview_path_filename = cxn.registry.get(self.args.registry_labview_path_key)
        except:
            raise Exception('Cannot read the LabRAD Registry. Please check that the Registry path ' + 
                  str([''] + self.args.registry_path) + ' and the key name ' + self.args.registry_labview_path_key + ' are correct.')
        
        print('Starting the DC Rack LabVIEW VI...')
        dc_rack_vi = os.path.join(LABRAD_PATH, DC_RACK_LABVIEW_VI_PATH, DC_RACK_LABVIEW_VI_FILENAME)
         
        if not os.path.isfile(dc_rack_vi):
            raise Exception('Cannot locate the DC Rack LabVIEW VI ' + dc_rack_vi  + '.')
        
        try:
            self.processes['DC Rack LabVIEW VI'] = sp.Popen('"' + labview_path_filename + '" "' + dc_rack_vi + '"')
        except OSError:
            raise Exception('Failed to start the DC Rack LabVIEW VI.')
        
        print('Please press "Run" button in the LabVIEW VI window.')
        self._waitTillEnterKeyIsPressed()

# Define a main() function
def main():
    with StartAndBringUp() as inst:
        inst.startLabRAD()
        for prog in inst.readProgramList():
            if prog.lower() in ['directethernet', 'direct ethernet']:
                inst.startDirectEthernetServer()
            elif prog.lower() in ['labradnode', 'labrad node']:
                inst.startLabRADNode()
            elif prog.lower() in ['labradnodeservers', 'labradnode servers', 'labrad node servers']:
                inst.startLabRADNodeServers()
            elif prog.lower() in ['ghzfpgabringup', 'ghz fpga bring up']:
                inst.bringUpGHzFPGAs()
            elif prog.lower() in ['dcracklabviewvi', 'dcrack labview vi', 'dc rack labview vi']: 
                inst.startDCRackLabVIEWVI()
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()