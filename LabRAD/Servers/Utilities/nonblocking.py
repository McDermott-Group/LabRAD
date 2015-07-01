from twisted.internet import reactor, defer

def sleep(time_in_seconds):
    """Use this function instead of time.sleep(time_in_seconds) in methods decorated with inlineCallbacks."""
    d = defer.Deferred()
    reactor.callLater(time_in_seconds, d.callback, None)
    return d