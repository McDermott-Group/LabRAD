'''
Run the LabRAD node service through python, rather than through twistd.

For this to work, the node must be installed as a twistd service;
see twisted/plugins/labrad_node.py

For documentation on the node, see the top of its source file:
labrad/node/__init__.py
'''

from twisted.scripts.twistd import run
from sys import argv
argv[1:] = ['-n', 'labradnode']
run()