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
This script can be used to start LabRAD servers with the LabRAD node. 
Run "labradnode_servers.py -h" in the command line for more information.
"""

import os
import sys
import argparse

import labrad as lr
 
def parseArguments():
    parser = argparse.ArgumentParser(description='Start LabRAD servers with the LabRAD node.')
    parser.add_argument('--registry-path', 
                        nargs='*',
                        default=['Start Server Lists', os.environ['COMPUTERNAME'].lower()],
                        help='path in the LabRAD Registry to the key containing the list of servers to run;' +
                        " root folder name ''" + ' must be omitted (default: "Start Server Lists" "%%COMPUTERNAME%%")')
    parser.add_argument('--registry-key', 
                        default='Start Server List',
                        help='start server list Registry key (default: "Start Server List")')
    parser.add_argument('--node-server-name', 
                        default='node ' + os.environ['COMPUTERNAME'].lower(),
                        help='LabRAD node server name (default: "node %%COMPUTERNAME%%"')
    return parser.parse_args()
 
def startServers(args):
    print('Connecting to LabRAD...')
    try:
        cxn = lr.connect()
    except:
        print('Could not connect to LabRAD. The LabRAD program does not appear to be running.')
        raise

    running_servers = [name for _, name in cxn.manager.servers()]
    if args.node_server_name not in running_servers:
        raise Exception("Could not connect to the LabRAD node server '" + args.node_server_name + "'. " + 
            "The server does not appear to be running.")

    print('Getting the list of servers from the LabRAD Registry...')
    try:
        cxn.registry.cd([''] + args.registry_path)
        server_list = cxn.registry.get(args.registry_key)
    except:
        print('Could not read the LabRAD Registry. Please check that the Registry path ' + 
              str([''] + args.registry_path) + ' and the key name ' + args.registry_key + ' are correct.') 
    
    # Go through and start all the servers that are not already running.
    print('Starting the servers...')
    for server in server_list:
        if server not in running_servers:
            try:
                cxn.servers[args.node_server_name].start(server)
            except Exception as e:
                raise Exception( 'Could not start ' + server + ': ' + str(e) + '.')
    cxn.disconnect()

def main():
    startServers(parseArguments())
   
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()