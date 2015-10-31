from twisted.internet import reactor, defer

import labrad.units as units

def sleep(time_in_seconds=1):
    """This is a non-blocking sleep function that could be used to delay 
    the code execution by a specified amount of time (given in seconds). 
    
    For example, to add a non-blocking ten-second delay to your code,
    you can add line "yield sleep(10)" to a method that is decorated 
    with "@inlineCallbacks".
    
    Input:
        time_in_seconds: either a number, e.g. "5" (less preferable), or
        a labrad.units.Value object, e.g. "10 * s" (preferable).
    """
    if isinstance(time_in_seconds, units.Value):
        time_in_seconds = time_in_seconds['s']
            
    d = defer.Deferred()
    reactor.callLater(time_in_seconds, d.callback, None)
    return d