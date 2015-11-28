# Calls to ghz_fpga server match v3.3.0 call signatures and outputs.

import argparse

import labrad

NUM_TRIES = 3

 
def _parse_arguments():
    parser = argparse.ArgumentParser(description='Automatically ' +
            'bring-up the FPGA GHz boards.')
    parser.add_argument('--password',
            default=None,
            help='LabRAD password')
    return parser.parse_args()

def bringup_board(fpga, board, optimizeSD=False, sdVal=None):
    """Bringup a single board connected to the given FPGA Server.
    
    Returns:
    (boardType, results, {'success0':bool, 'success1':bool})
    """
    fpga.select_device(board)
    
    # Determine if board is ADC. If so, bringup ADC.
    if board in fpga.list_adcs():
        fpga.adc_bringup()
        return ('ADC', None, {})
    
    if board in fpga.list_dacs():
        if sdVal is None:
            resp = fpga.dac_bringup(optimizeSD)
        else:
            resp = fpga.dac_bringup(False, sdVal)
        results = {}
        okay = []
        lvdsOkay = []
        for dacdata in resp:
            dacdict = dict(dacdata)
            dac = dacdict.pop('dac')
            results[dac] = dacdict
            lvdsOkay.append(dacdict['lvdsSuccess'])
            okay.append(dacdict['fifoSuccess'])
            okay.append(dacdict['bistSuccess'])
            
        return ('DAC', results, {'bistFIFO': all(okay),
                                 'lvds': all(lvdsOkay)})
    
def bringup_boards(fpga, boards):
    """
    Bringup a list of boards and return the Bist/FIFO and lvds
    success for each one.
    
    Returns:
    dictionary of the form {<board name> : {<item name> : bool,...},...}
    where the items are 'lvds', 'FIFO', etc, and the boolean value
    is True for successful bringup of that item and False for failed
    bringup of that item.
    """
    successes = {}
    tries = {}
    for board in boards:
        # Temporarily set a value to False so the while loop will run at
        # least once.
        successDict = {0: False}
        k = 0
        while k < NUM_TRIES and not all(successDict.values()):
            k += 1
            tries[board] = k
            boardType, result, successDict = bringup_board(fpga, board)
            successes[board] = successDict

    failures = [board for board in boards
                if not all(successes[board].values())]

    return successes, failures, tries

def auto_bringup(fpga):
    """
    Bringup all boards.
    
    Returns:
    tuple  of the form, ([<failed board name>,...], {<board name> :
    <number of bring-up attempts>,...}). If the first item of the list,
    which is a list, is empty, then all boards have been succesfully
    brought-up.
    """
    boards = fpga.list_devices()
    successes, failures, tries = bringup_boards(fpga,
            [board[1] for board in boards])
    return successes, failures, tries

def main():
    args = _parse_arguments()
    with labrad.connect(password=args.password) as cxn:
        fpga = cxn['ghz_fpgas']
        no_success = True
        while no_success:
            successes, failures, tries = auto_bringup(fpga)
            successes = [board for board in successes.keys()
                      if board not in failures]
            if successes:
                print('The following boards have been succesfully ' +
                        'brought up:')
                for key in successes:
                    print(key + ' (attempts: ' + str(tries[key]) + ')')
            if failures:
                print('The following boards failed:')
                for key in failures:
                    print(key)
            else:
                no_success = False


if __name__ == '__main__':
    main()