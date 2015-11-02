from twisted.internet import reactor, defer

import labrad.units as units

def sleep(time=1):
    """This is a non-blocking sleep function that could be used to delay 
    the code execution by a specified amount of time (given in seconds). 
    
    For example, to add a non-blocking ten-second delay to your code,
    you write "yield sleep(10)" in your method that is decorated 
    with "@inlineCallbacks".
    
    Input:
        time: time to sleep in seconds. It should be specified either
            as a number, e.g. "5" (less preferable), or as
            a labrad.units.Value object, e.g. "10 * s" (preferable).
    """
    if isinstance(time, units.Value):
        time = time['s']
            
    d = defer.Deferred()
    reactor.callLater(time, d.callback, None)
    return d