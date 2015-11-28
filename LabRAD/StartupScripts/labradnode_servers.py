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
Run "labradnode_servers.py -h" in the command line for the command line
input options.
"""

import os
import sys
import argparse

import labrad as lr
from labrad.server import inlineCallbacks
 
def parseArguments():
    parser = argparse.ArgumentParser(description='Start LabRAD ' +
            'servers with the LabRAD node.')
    parser.add_argument('--registry-path', 
            nargs='*',
            default=['Start Lists', os.environ['COMPUTERNAME'].lower()],
            help='path in the LabRAD Registry to the key containing ' +
            'the list of servers to run;' + " root folder name '' " + 
            'must be omitted (default: "Start Lists" "%COMPUTERNAME%")')
    parser.add_argument('--registry-start-list-key', 
            default='Start Server List',
            help='Registry key containg the list of servers to run ' +
            '(default: "Start Server List")')
    parser.add_argument('--node-name', 
            default='node ' + os.environ['COMPUTERNAME'].lower(),
            help='LabRAD node name (default: "node %%COMPUTERNAME%%"')
    parser.add_argument('--password',
            default=None,
            help='LabRAD password')
    return parser.parse_args()

@inlineCallbacks    
def startServers(args):
    print('Connecting to LabRAD...')
    try:
        cxn = yield lr.connect(password=args.password)
    except:
        raise Exception('Could not connect to LabRAD. The LabRAD ' +
                'program does not appear to be running.')

    running_servers = [name for id, name in cxn.manager.servers()]
    if args.node_name not in running_servers:
        raise Exception("Cannot connect to the LabRAD node server '" +
                args.node_name + "'. " + 
                "The server does not appear to be running.")

    print('Getting the list of servers from the LabRAD Registry...')
    try:
        yield cxn.registry.cd([''] + args.registry_path)
        server_list = yield cxn.registry.get(args.registry_start_list_key)
    except:
        raise Exception('Cannot read the LabRAD Registry. Please ' +
                'check that the Registry path ' + 
                str([''] + args.registry_path) + ' and the key name ' +
                args.registry_key + ' are correct.')
    
    # Go through and start all the servers that are not already running.
    print('Starting the servers...')
    for server in server_list:
        if (server not in running_servers and
                (os.environ['COMPUTERNAME'].lower() + ' ' + server) 
                not in running_servers):
            try:
                yield cxn.servers[args.node_name].start(server)
            except Exception as e:
                raise Exception( 'Could not start ' + server + ': ' +
                    str(e) + '.')
    yield cxn.disconnect()

def main():
    startServers(parseArguments())
   
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()