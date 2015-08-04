import pyle.dataking.util as dataUtil
CONFIG_KEY = raw_input('CONFIG_KEY: ')


def loadDacsAdcs(sample):
    sample, devices = dataUtil.loadDevices(sample, configKey=CONFIG_KEY)
    dacs = []
    adcs = []
    for d in devices.keys():
        if 'DAC' in d:
            dacs.append(devices[d])
        elif 'ADC' in d:
            adcs.append(devices[d])
        else:
            raise Exception('Device must be DAC or ADC')
    return dacs, adcs
    
def boardName2Info(boardName):
    """Gets the (number, type) of an FPGA board from the name"""
    pattern = '\d+' #match any number of digits
    p = re.compile(pattern)
    m = p.search(boardName)
    sp = m.span()
    boardNumber = int(boardName[sp[0]:sp[1]])
    if 'DAC' in boardName:
        boardType = 'DAC'
    elif 'ADC' in boardName:
        boardType = 'ADC'
    else:
        raise Exception('Board type of board %s not recognized' %boardName)
    return (boardNumber, boardType)