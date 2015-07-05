from twisted.internet import reactor, defer

def sleep(time_in_seconds=1):
    """This is a non-blocking function that could be used to delay the code execution 
    by specified amount of time expressed in seconds. 
    
    For example, to add a non-blocking ten-second delay to your code, you can write `yield sleep(10)` 
    inside a method that is decorated with `@inlineCallbacks`.
    """
    d = defer.Deferred()
    reactor.callLater(time_in_seconds, d.callback, None)
    return d
