from twisted.internet import reactor, defer

def sleep(time_in_seconds):
    d = defer.Deferred()
    reactor.callLater(time_in_seconds, d.callback, None)
    return d