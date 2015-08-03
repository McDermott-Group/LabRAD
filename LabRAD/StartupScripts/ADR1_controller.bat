START "" ../LabRAD-v1.1.4.exe
timeout 3
START "" twistd.py -n labradnode
echo Please enter password in node window that pops up
pause
START "" ../Servers/ADR/ADRServer.py -a ADR1 -w ""
START "" ../Servers/ADR/ADRClient.py -w ""
