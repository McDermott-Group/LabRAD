# McDermott-Group
This repository will hold much of the software used and/or created by the McDermott Lab at UW Madison (Physics).  Proper credit should go to those in the Martinis group at UCSB for their development of LabRAD, which forms the base of much of our code here.  Indeed, many of these files were taken directly from their code, and sometimes edited slightly to fit our needs.

INSTALLATION:

Notes: 
Installation order matters.  Do it in this order.
If you are installing from a directory, sometimes errors are thrown if the path length is too long. :/

python-2.7.9.msi - make sure that add python to path is selected
numpy-1.9.1-win32-superpack-python2.7.exe
scipy-0.15.1-win32-superpack-python2.7.exe
run in command line: "pip2.7 install ipython"
run in command line: "pip2.7 install matplotlib"
run in command line: "pip2.7 install twisted"
pywin32-219.win32-py2.7.exe
run in command line: "pip2.7 install pyserial"
run in command line: "pip2.7 install pyvisa"
pylabrad-0.92.5.win32-py2.7.exe
replace "C:\Python27\Lib\site-packages\labrad\types.py" with types.py from this folder
