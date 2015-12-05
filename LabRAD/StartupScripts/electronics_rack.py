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
This script can be used to start LabRAD manager, servers with the LabRAD
node and some other programs. Run "labrad_servers_clients.py -h" in
the command line for the command line input options.
"""

import os
import sys
import time
import subprocess as sp
import argparse
from msvcrt import getch, kbhit

import labrad


# Absolute path to the folder with LabRAD.
LABRAD_PATH = os.path.join(os.environ['HOME'],
        r'Desktop\Git Repositories\LabRAD\LabRAD')

# SCRIPT_PATH = os.path.dirname(__file__)
# LABRAD_PATH = os.path.join(SCRIPT_PATH.rsplit('LabRAD', 1)[0], 'LabRAD')

LABRAD_INI = 'LabRAD.ini'

# Python pythonw.exe executable that can be used to start python
# GUIs without the associated terminal window.
PYTHONW = sys.executable.replace('python.exe', 'pythonw.exe')

# Relative paths with respect to LABRAD_PATH.
DIRECT_ETHERNET_SERVER_PATH = r'Servers\DirectEthernet'
LABRAD_NODE_PATH = r'StartupScripts'
LABRAD_NODE_SERVERS_PATH = r'StartupScripts'
GHZ_FPGA_BRING_UP_PATH = r'Servers\Instruments\GHzBoards'
DC_RACK_GUI_PATH = r'Servers\Instruments\DCRack'
ADR_PATH = r'Servers\ADR'

# Corresponding file names with the extensions.
LABRAD_FILENAME = 'LabRAD-v1.1.4.exe'
LABRAD_NODE_FILENAME = 'labradnode.py'
LABRAD_NODE_SERVERS_FILENAME = 'labradnode_servers.py'
DIRECT_ETHERNET_SERVER_FILENAME = 'DirectEthernet.exe'
GHZ_FPGA_BRING_UP_FILENAME = 'auto_ghz_fpga_bringup.py'
DC_RACK_GUI_FILENAME = 'DCRackGUI.pyw'
ADR_CLIENT_FILENAME = 'ADRClient.py'

class QuitException(Exception): pass


class StartAndBringUp:
    def __init__(self):
        if not os.path.exists(LABRAD_PATH):
            raise Exception('LABRAD_PATH = ' + LABRAD_PATH +
                    ' does not exist.')
        self.processes = {}
        self.args = self._parseArguments()
        self._password = self.args.password
 
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if hasattr(self, '_cxn'):
            self._cxn.disconnect()
        if exception_type is not None:
            print('Closing the started processes...')
            for process in self.processes:
                return_code = self.processes[process].poll()
                if return_code is None:
                    self.processes[process].terminate()
        if exception_type == QuitException:
            return True
            
    def _parseArguments(self):
        parser = argparse.ArgumentParser(description='Start LabRAD, ' +
                'LabRAD servers and other programs.')
        parser.add_argument('--registry-path', 
                nargs='*',
                default=['Start Lists', os.environ['COMPUTERNAME'].lower()],
                help='path in the LabRAD registry to the key ' +
                'containing the list of programs to run;' +
                " root folder name ''" + ' must be omitted ' +
                '(default: "Start Lists" "%COMPUTERNAME%")')
        parser.add_argument('--registry-start-program-key', 
                default='Start Program List',
                help='Registry key containing the list of programs ' +
                'to run (default: "Start Program List")')
        parser.add_argument('--registry-start-server-key', 
                default='Start Server List',
                help='Registry key containg the list of servers to run ' +
                '(default: "Start Server List")')
        parser.add_argument('--registry-labrad-host-key',
                default='LabRAD Host',
                help='Regitry key with the value of %LabRADHost% ' +
                'environment variable (default: "LabRAD Host")')
        parser.add_argument('--registry-labrad-port-key',
                default='LabRAD Port',
                help='Resistry key with the value of %LabRADPort% ' +
                'environment variable as a string (default: "LabRAD Port")')
        parser.add_argument('--registry-labrad-node-key',
                default='LabRAD Node',
                help='Resistry key with thevalue of %LabRADNode% ' +
                'environment variable (default: "LabRAD Node")')
        parser.add_argument('--password',
                default=None,
                help='LabRAD password')
        parser.add_argument('--adr',
                default='ADR3',
                help='ADR to control')
        return parser.parse_args()  
    
    def _LabRADConnect(self):
        if not hasattr(self, '_cxn'):
            print('Connecting to LabRAD...')
            try:
                self._cxn = labrad.connect(password=self._password)
            except:
                raise Exception('Could not connect to LabRAD. The LabRAD' +
                        'manager does not appear to be running.')
                
    def _changeResitryPath(self):
        self._LabRADConnect()
        try:
            print('Changing the LabRAD registry directory...')
            self._cxn.registry.cd([''] + self.args.registry_path)
        except:
            raise Exception('Could not read the LabRAD registry. ' +
                    'Please check that the AFS is on and the registry path ' +
                    str([''] + self.args.registry_path) + ' is correct.')
    
    def _waitTillEnterKeyIsPressed(self):
        while kbhit():
            getch()
        print('\n\t[ENTER]: Continue.\n\t[Q]:\t Quit and close ' +
                'the started processes.\n')
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
        # Open the LabRAD initialization file to read the password.
        labrad_ini_file = os.path.join(LABRAD_PATH, LABRAD_INI)
        if os.path.isfile(labrad_ini_file):
            with open(labrad_ini_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                if line.find('Password: ') != -1:
                    self._password = line.split('Password: ')[-1]
                    break

        try:
            self._cxn = labrad.connect(password=self._password)
        except:
            print('Starting the LabRAD manager...')
            labrad_filename = os.path.join(LABRAD_PATH, LABRAD_FILENAME)
            if not os.path.isfile(labrad_filename):
                raise Exception('Could not locate the LabRAD ' +
                        'executable file ' + labrad_filename + '.')
            try:
                self.processes['LabRAD'] = sp.Popen(labrad_filename)
            except OSError:
                raise Exception('Failed to start the LabRAD manager.')
            print('Please press [Run server] button in the LabRAD ' +
                    'window if it has not started automatically.')
            if self._password is None:
                self._waitTillEnterKeyIsPressed()
            else:
                time.sleep(3)

        try:
            os.environ['LabRADPassword'] = self._password
            print("Environment variable %LabRADPassword% is set to '" +
                    os.environ['LabRADPassword'] + "'.\n")
        except:
            pass

    def readRegistry(self):
        print('Getting the list of programs and servers to run from' +
                ' the LabRAD registry...')
        self._changeResitryPath()
        try:
            print('Getting the list of programs to run...')
            self.program_list = \
                self._cxn.registry.get(self.args.registry_start_program_key)
        except:
            raise Exception('Could not read the LabRAD registry. ' +
                    'Please check that the registry key name ' + 
                    self.args.registry_start_program_key + ' is correct.')
        try:
            os.environ['LabRADHost'] = \
                self._cxn.registry.get(self.args.registry_labrad_host_key)
            print("Environment variable %LabRADHost% is set to '" +
                    str(os.environ['LabRADHost']) + "'.")
        except:
            pass
        try:
            os.environ['LabRADPort'] = \
                self._cxn.registry.get(self.args.registry_labrad_port_key)
            print("Environment variable %LabRADPort% is set to '" +
                    str(os.environ['LabRADPort']) + "'.")
        except:
            pass
        try:
            os.environ['LabRADNode'] = \
                self._cxn.registry.get(self.args.registry_labrad_node_key)
        except:
            os.environ['LabRADNode'] = os.environ['COMPUTERNAME'].lower()
        print("Environment variable %LabRADNode% is set to '" +
                os.environ['LabRADNode'] + "'.\n")
   
    def getProgramList(self):
        if hasattr(self, 'program_list'):
            return self.program_list
        else:
            return []

    def startLabRADNode(self):
        self._LabRADConnect()
        running_servers = [name for id, name in self._cxn.manager.servers()]
        if 'LabRADNode' in os.environ:
            node_name = 'node ' + os.environ['LabRADNode']
        else:
            node_name = 'node ' + os.environ['COMPUTERNAME'].lower()
        if node_name not in running_servers:
            print('Starting the LabRAD node...')
            node_filename = os.path.join(LABRAD_PATH, LABRAD_NODE_PATH,
                    LABRAD_NODE_FILENAME)
            try:
                self.processes['LabRAD Node'] = sp.Popen([sys.executable,
                        node_filename], creationflags=sp.CREATE_NEW_CONSOLE)
            except OSError:
                raise Exception('Failed to start the LabRAD node.')
            print('Please enter the password in the LabRAD node ' +
                    'window that poped up.')
            print('Do not close the window when you are done.')
            if self._password is None or self._password == '':
                self._waitTillEnterKeyIsPressed()
            else:
                time.sleep(2)
                print('The LabRAD node has been started.\n')
        
    def startLabRADNodeServers(self):
        print('Starting the required servers with the LabRAD node...')
        node_servers_filename = os.path.join(LABRAD_PATH,
                LABRAD_NODE_SERVERS_PATH, LABRAD_NODE_SERVERS_FILENAME)
        try:
            self.processes['LabRAD Node Servers'] = sp.Popen([sys.executable,
                    node_servers_filename, '--registry-start-server-key',
                    self.args.registry_start_server_key,
                    '--password', self._password])
        except:
            raise Exception('Failed to connect to the LabRAD node.')
        self.processes['LabRAD Node Servers'].wait()
        print('The servers have been started.\n')
        
    def startDirectEthernetServer(self):
        self._LabRADConnect()
        server_name = os.environ['LabRADNode'] + ' Direct Ethernet'
        running_servers = [name for id, name in self._cxn.manager.servers()]
        if server_name not in running_servers:
            print('Starting the Direct Ethernet server...')
            direct_ethernet = os.path.join(LABRAD_PATH,
                    DIRECT_ETHERNET_SERVER_PATH,
                    DIRECT_ETHERNET_SERVER_FILENAME)
            if not os.path.isfile(direct_ethernet):
                raise Exception('Could not locate the Direct Ethernet' +
                        ' Server sys.executable file ' +
                        direct_ethernet + '.')
            try:
                self.processes['Direct Ethernet Server'] = sp.Popen(direct_ethernet)
            except OSError:
                raise Exception('Failed to start Direct Ethernet Server.') 
            print('If prompted, please specify in the Direct Ethernet' +
                  ' window the LabRAD host name, port, password,' +
                  ' and/or node name.')
            if self._password is None or self._password == '':
                self._waitTillEnterKeyIsPressed()
            else:
                time.sleep(1)
                print('The Direct Ethernet Server has been started.\n')
        
    def bringUpGHzFPGAs(self):
        print('Bringing up the GHz FPGA boards...')
        bring_up = os.path.join(LABRAD_PATH, GHZ_FPGA_BRING_UP_PATH,
                GHZ_FPGA_BRING_UP_FILENAME)
        if not os.path.isfile(bring_up):
            raise Exception('Could not locate the GHz FPGA bring-up ' +
                    'script ' + bring_up + '.')
        try:
            self.processes['GHz FPGA Bring Up'] = sp.Popen([sys.executable,
                    bring_up, '--password', self._password])
        except OSError:
            raise Exception('Failed to start the GHz FPGA bring up script.')
        self.processes['GHz FPGA Bring Up'].wait()
        print('The GHz FPGA boards have been brought up.\n')
        
    def startDCRackGUI(self):
        print('Openning the DC Rack GUI...')
        gui_path = os.path.join(LABRAD_PATH, DC_RACK_GUI_PATH)
        gui_file = os.path.join(gui_path, DC_RACK_GUI_FILENAME)
        if not os.path.isfile(gui_file):
            raise Exception('Could not locate the DC Rack GUI script ' +
                    gui_file + '.')
        try:
            self.processes['DC Rack GUI'] = sp.Popen([PYTHONW,
                    gui_file, '--password', self._password],
                    creationflags=sp.CREATE_NEW_CONSOLE, cwd=gui_path)
        except OSError:
            raise Exception('Failed to start the DC Rack GUI.')
        print('The DC Rack GUI has been opened.')
        self._waitTillEnterKeyIsPressed()
        
    def startADRClient(self):
        print('Openning the ADR client GUI...')
        gui_path = os.path.join(LABRAD_PATH, ADR_PATH)
        gui_file = os.path.join(gui_path, ADR_CLIENT_FILENAME)
        if not os.path.isfile(gui_file):
            raise Exception('Could not locate the ADR client GUI script ' +
                    gui_file + '.')
        try:
            self.processes['ADR client GUI'] = sp.Popen([sys.executable,
                    gui_file, '-w', self._password],
                    creationflags=sp.CREATE_NEW_CONSOLE, cwd=gui_path)
        except OSError:
            raise Exception('Failed to start the ADR client GUI.')
        print('The ADR client GUI has been opened.')
        self._waitTillEnterKeyIsPressed()

def main():
    with StartAndBringUp() as inst:
        inst.startLabRAD()
        inst.readRegistry()
        progs = inst.getProgramList()
        progs = [prog.lower().replace(' ', '').replace('-', '')
                for prog in progs]
        for prog in progs:
            if prog == 'directethernet':
                inst.startDirectEthernetServer()
                break
        for prog in progs:
            if prog =='labradnode':
                inst.startLabRADNode()
            elif prog == 'labradnodeservers':
                inst.startLabRADNodeServers()
            elif prog == 'adrclient':
                inst.startADRClient()
            elif prog == 'ghzfpgabringup':
                inst.bringUpGHzFPGAs()
            elif prog == 'dcrackgui': 
                inst.startDCRackGUI()


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()