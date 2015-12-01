import os
if __file__ in [f for f in os.listdir('.') if os.path.isfile(f)]:
    # This is executed when the script is loaded by the labradnode.
    SCRIPT_PATH = os.path.dirname(os.getcwd())
else:
    # This is executed if the script is started by clicking or
    # from a command line.
    SCRIPT_PATH = os.path.dirname(__file__)
LABRAD_PATH = os.path.join(SCRIPT_PATH.rsplit('LabRAD', 1)[0])
import sys
if LABRAD_PATH not in sys.path:
    sys.path.append(LABRAD_PATH)

import labrad

from LabRAD.TestScripts.fpgaTest.pyle.pyle.workflow import switchSession as pss #(P)yle(S)witch(S)ession
from LabRAD.TestScripts.fpgaTest.pyle.pyle.util import sweeptools as st
import fpgaTest

def switchSession(session=None, user=None):
    """Switch the current session, using the global connection object."""
    global s
    if user is None:
        user = s._dir[1]
    s = pss(cxn, user, session, useDataVault=False)

# connect to labrad and setup a wrapper for the current sample
cxn = labrad.connect()
switchSession(user='TestUser')

fpga = cxn.ghz_fpgas

# print(str(fpgaTest.daisyCheck(s, cxn, 10, 10, True)))
#fpgaTest.dacSignal(s, fpga)
#fpgaTest.runAverage(s, fpga, plot=True)
fpgaTest.average(s, cxn, 60, plot=True, save=False)
#fpgaTest.sumCheck(s, cxn, plot=True, save=False)
#fpgaTest.spectrum(s, cxn, plot=True, save=False)
#fpgaTest.sideband(s, cxn, plot=True, save=False)
# fpgaTest.filterCompare(s, cxn, [('square', 0), ('hann', 0), 
        # ('gaussian', 10)], plot=True, save=False)
#fpgaTest.phase(s, cxn, 'DAC', dacAmp=0.25, plot=True, save=False)
#fpgaTest.dacAmpToAdcAverage(s, cxn, plot=True, save=False)
#fpgaTest.adcAmpToVoltage(s, cxn, plot=True, save=False)