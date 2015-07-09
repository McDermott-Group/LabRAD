"""
Version Info
version = 3.0
server: ghz_fpgas
server version: 3.3.0
"""

# Calls to ghz_fpga server match v3.3.0 call signatures and outputs.

from __future__ import with_statement

import random
import time

import labrad
from math import sin, pi
from msvcrt import getch, kbhit

FPGA_SERVER='ghz_fpgas'
DACS = ['A','B']
NUM_TRIES = 2

def bringupBoard(fpga, board, printOutput=True, fullOutput=False, optimizeSD=False, sdVal=None):
    """Bringup a single board connected to the given fpga server.
    
    RETURNS
    (boardType, results, {'success0':bool, 'success1':bool})
    """
    fpga.select_device(board)
    
    # Determine if board is ADC. If so, bringup ADC.
    if board in fpga.list_adcs():
        fpga.adc_bringup()
        print ''
        print '%s ok' %board
        return ('ADC', None, {})
    
    if board in fpga.list_dacs():
        if sdVal is None:
            resp = fpga.dac_bringup(optimizeSD)
        else:
            resp = fpga.dac_bringup(False,sdVal)
        results={}
        okay = []
        lvdsOkay = []
        for dacdata in resp:
            dacdict=dict(dacdata)
            dac = dacdict.pop('dac')
            results[dac] = dacdict
            
            if printOutput:
                print ''
                print 'DAC %s LVDS Parameters:' % dac
                print '  SD: %d' % dacdict['lvdsSD']
                print '  Check: %d' % dacdict['lvdsCheck']
                print '  Plot MSD:  ' + ''.join('_-'[ind] for ind in dacdict['lvdsTiming'][1])
                print '  Plot MHD:  ' + ''.join('_-'[ind] for ind in dacdict['lvdsTiming'][2])
            lvdsOkay.append(dacdict['lvdsSuccess'])
            
            if printOutput:
                print ''
                print 'DAC %s FIFO Parameters:' % dac
                print '  FIFO calibration had to run %d times' %dacdict['fifoTries']
                if dacdict['fifoSuccess']:
                    print '  FIFO PHOF:   %d' % dacdict['fifoPHOF']
                    print '  Clk Polarity:  %d' % dacdict['fifoClockPolarity']
                    print '  FIFO Counter:  %d (should be 3)' %dacdict['fifoCounter']
                else:
                    print '  FIFO Failure!'
            okay.append(dacdict['fifoSuccess'])
            
            if printOutput:
                print ''
                print 'DAC %s BIST:' % dac
                print '  Success:' + yesNo(dacdict['bistSuccess'])
            okay.append(dacdict['bistSuccess'])
            
        print  ''
        if all(okay):
            print '%s ok' %board
            if not all(lvdsOkay):
                print 'LVDS warning'
        else:
            print '%s Bringup Failure!!! Reinitialize bringup!!!' %board
            
        if fullOutput:
            return ('DAC', results, {'bistFIFO':all(okay), 'lvds':all(lvdsOkay)})
        else:
            return ('DAC', None, {})


def yesNo(booleanVal):
    if booleanVal:
        return 'Yes'
    elif (not booleanVal):
        return 'No'
    else:
        raise Exception


def checkBoard(fpga, board, checkLock=False):
    """Bringup a single board, capturing exceptions and returning a string status message."""
    print 'Bringing up %s...' % board
    
    if checkLock:
        print 'Checking lock...',
        fpga.select_device(board)
        if not fpga.pll_query(): #pll_query() returns True if PLL has lost lock
            print 'OK'
            print
            return 'ok'
        else:
            print 'UNLOCKED'
    
    try:
        result = bringupBoard(fpga, board)
    except Exception, e:
        print e
        result = e # a problem happened
    return result


def getChoice(keys):
    """Get a keypress from the user from the specified keys."""
    while kbhit():
        getch()
    r = ''
    while not r or r not in keys:
        r = raw_input().upper()
    return r

def bringupBoardGroup(fpga, boardGroup):
    boards = [brd[1] for brd in fpga.list_devices(boardGroup)]
    successes = bringupBoards(fpga, boards)
    return successes
    
def bringupBoards(fpga, boards, noisy=False):
    """Bringup a list of boards and return the Bist/FIFO and lvds success for
    each one.
    
    RETURNS
    dictionary of the form {<board name>:{<item name>:bool,...},...}
    where the items are 'lvds', 'FIFO', etc, and the boolean value
    is True for successful bringup of that item and False for failed
    bringup of that item.
    
    TODO:
    Put in code to keep track of retries and report this to the user
    """
    successes = {}
    triesDict = {}
    for board in boards:
        successDict={0:False} #Temporarily set a value to False so while loop will run at least once
        tries=0
        if noisy:
            print 'bringup up %s' %board
        while tries<NUM_TRIES and not all(successDict.values()):
            tries += 1
            triesDict[board]=tries
            boardType,result,successDict = bringupBoard(fpga, board, fullOutput=True,
                                                            printOutput=False)
            successes[board] = successDict
        print
        print
    failures = []
    
    for board in boards:
        if triesDict[board]>1:
            print 'WARNING: Board %s took %d tries to succeed' %(board,triesDict[board])
        if not all(successes[board].values()):
            failures.append(board)
    if not failures:
        print 'All boards successful!'
    else:
        print 'The following boards failed:'
        for board in failures:
            print board
    return successes

def allBringups(fpga,boards):
    """JJW
    This function should be replaced by bringupBoards
    """
    raise Exception('Depricated. Use bringupBoards instead')
    dacFails = []
    dacIssues = {}
    for boardSelection in boards:
        board = boardSelection[1]
        print 'Bringing up %s...' % board
        try:
            boardType, results, successDict = bringupBoard(fpga, board,
                                                           fullOutput=True,
                                                           printOutput=False)
        except Exception, e:
            dacFails.append(board)
            print e
        if boardType == 'DAC':
            tries = 1
            while (tries < NUM_TRIES) and (not all()):
                boardType, results, successDict = bringupBoard(fpga, board,
                                            fullOutput=True, printOutput=False)
                tries += 1
                dacIssues[board] = tries
            if not okay:
                dacIssues.pop(board)
                dacFails.append(board)
    print
    print
    print
    if len(dacIssues)>0:
        print 'The following boards took more than one try to bring up:'
        for key in dacIssues:
            print key + ' took %d tries' %dacIssues[key]
    if len(dacFails)>0:
        print 'All boards brought up except:'
        print dacFails
    else:
        print 'All boards successful!!!  :-)'
        


def interactiveBringup(fpga, board):
    boardType = checkBoard(fpga, board)

    if boardType[0] == 'DAC':
        ccset = 0
        while True:
            print
            print
            print 'Choose:'
            print
            print '  [1] : Output 0x0000s'
            print '  [2] : Output 0x1FFFs'
            print '  [3] : Output 0x2000s'
            print '  [4] : Output 0x3FFFs'
            print
            print '  [5] : Output 100MHz sine wave'
            print '  [6] : Output 200MHz sine wave'
            print '  [7] : Output 100MHz and 175MHz sine wave'
            print '  [8] : Output 100 ns squared wave'
            print
            print '        Current Cross Controller Setting: %d' % ccset
            print '  [+] : Increase Cross Controller Adjustment by 1'
            print '  [-] : Decrease Cross Controller Adjustment by 1'
            print '  [*] : Increase Cross Controller Adjustment by 10'
            print '  [/] : Decrease Cross Controller Adjustment by 10'
            print
            print '  [I] : Reinitialize'
            print
            print '  [Q] : Quit'
    
            k = getChoice('12345678+-*/IQ')
    
            # run various debug sequences
            if k in '12345678':
                if k == '1': fpga.dac_debug_output(0xF0000000, 0, 0, 0)
                if k == '2': fpga.dac_debug_output(0xF7FFDFFF, 0x07FFDFFF, 0x07FFDFFF, 0x07FFDFFF)
                if k == '3': fpga.dac_debug_output(0xF8002000, 0x08002000, 0x08002000, 0x08002000)
                if k == '4': fpga.dac_debug_output(0xFFFFFFFF, 0x0FFFFFFF, 0x0FFFFFFF, 0x0FFFFFFF)
    
                def makeSines(freqs, T):
                    """Build sram sequence consisting of superposed sine waves."""
                    wave = [sum(sin(2*pi*t*f) for f in freqs) for t in range(T)]
                    sram = [(long(round(0x1FFF*y/len(freqs))) & 0x3FFF)*0x4001 for y in wave]
                    sram[0] = 0xF0000000 # add triggers at the beginning
                    return sram
    
                def makePulse(delay, length):
                    """Build sram sequence for a pulse."""
                    wave = [-1. for t in range(delay)] + [1. for t in range(length)]
                    sram = [(long(round(0x1FFF*y)) & 0x3FFF)*0x4001 for y in wave]
                    sram[0] = 0xF0000000 # add triggers at the beginning
                    return sram
    
                if k == '5': fpga.dac_run_sram(makeSines([0.100], 40), True)    
                if k == '6': fpga.dac_run_sram(makeSines([0.200], 40), True)    
                if k == '7': fpga.dac_run_sram(makeSines([0.100, 0.175], 40), True)
                if k == '8': fpga.dac_run_sram(makePulse(50, 50), True)
                print 'Running...'
    
            if k in '+-*/':
                if k == '+': ccset += 1
                if k == '-': ccset -= 1
                if k == '*': ccset += 10
                if k == '/': ccset -= 10
            
                if ccset > +63: ccset = +63
                if ccset < -63: ccset = -63
                fpga.dac_cross_controller('A', ccset)
                fpga.dac_cross_controller('B', ccset)
    
            if k == 'I': checkBoard(fpga, board)
            if k == 'Q': break

def selectFromList(options, title):
    print
    print
    print 'Select %s:' % title
    print
    keys = {}
    for i, opt in enumerate(options):
        key = '%d' % (i+1)
        keys[key] = opt
        print '  [%s] : %s' % (key, opt)
    keys['A'] = 'All'
    print '  [A] : All'
    keys['Q'] = None
    print '  [Q] : Quit'
    
    k = getChoice(keys)
    
    return keys[k]

def selectBoardGroup(fpga):
    groups = fpga.list_board_groups()
    group = selectFromList(groups, 'Board Group')
    return group

def doBringup():
    with labrad.connect() as cxn:
        fpga = cxn[FPGA_SERVER]
        group = selectBoardGroup(fpga)
        while True:
            if group is None:
                break
            elif group in fpga.list_board_groups():
                boards = fpga.list_devices(group)
            else:
                boards = fpga.list_devices()
            boardSelect = selectFromList(boards, 'FPGA Board')
            if boardSelect is None:
                break
            elif boardSelect == 'All':
                bringupBoards(fpga,[board[1] for board in boards],noisy=True)
            else:
                board = boardSelect[1]
                interactiveBringup(fpga, board)

if __name__ == '__main__':
    doBringup()