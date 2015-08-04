import labrad
import numpy as np
import time

ETHERNET_MAX_PKT_LEN = 1500 #octets

def packetTimer(de, port, src, dst, pktLen, numPkts):
    p = de.packet()
    p.connect(port)
    p.source_mac(src)
    p.destination_mac(dst)
    p.require_source_mac(src)
    p.require_destination_mac(dst)
    p.timeout(5)
    p.listen()
    p.send()
    #Build list of packets to send
    pkt = '0'*pktLen
    #Write packets out over the wire
    for _ in range(numPkts):
        de.write(pkt)
    de.collect(numPkts)
    time.sleep(1)
    #start timer
    tStart = time.clock()
    result = de.read(numPkts)
    tEnd = time.clock()
    return result, tEnd-tStart

    
def analyzePacketTiming(de, port, src, dst, N, pktLengths):
    times = np.array([])
    for pktLength in pktLengths:
        if N%pktLength != 0:
            raise Exception('pktLength must divide N')
        if pktLength > ETHERNET_MAX_PKT_LEN:
            raise Exception('Ethernet packets cannot exceed length %d' %ETHERNET_MAX_PKT_LEN)
    for pktLength in pktLengths[::-1]:
        numPkts = N/pktLength
        print 'Testing with %d packets of length %d' %(numPkts, pktLength)
        result, t = packetTimer(de, port, src, dst, pktLength, int(numPkts))
        times = np.hstack((times,t))
    return np.vstack((pktLengths, times[::-1])).T