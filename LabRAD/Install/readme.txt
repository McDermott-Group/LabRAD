Installation order matters. Follow the order described in this file. 
If you are installing from a directory, sometimes errors are thrown if the path length is too long. :/

Install Python and the required modules in this order (on a computer with an Internet access):
python-2.7.9.msi - make sure to select "add python to path"
numpy-1.9.2-win32-superpack-python2.7.exe
scipy-0.15.1-win32-superpack-python2.7.exe
run in the command line: "pip2.7 install ipython"
run in the command line: "pip2.7 install matplotlib"
VCForPython27.msi
run in the command line: "pip2.7 install twisted"
pywin32-219.win32-py2.7.exe
run in the command line: "pip2.7 install pyserial"
run in the command line: "pip2.7 install pyvisa"
pylabrad-0.92.5.win32-py2.7.exe
replace "C:\Python27\Lib\site-packages\labrad\types.py" with "types.py" from this folder
replace "C:\Python27\Lib\site-packages\labrad\gpib.py" with "gpib.py" from this folder
make sure tha "support.py" is present in "C:\Python27\Lib\site-packages\labrad\", if not, add it there from this folder

===================================================================================================
To update a package run "pip2.7 install --upgrade package_name" in the command line.

===================================================================================================
Below are some of the tested configurations. You can get this list by running "pip2.7 freeze" in the command line.

[dr2]
PyVISA==1.7
Twisted==15.2.0
enum34==1.0.4
ipython==3.1.0
matplotlib==1.4.3
numpy==1.9.2
pylabrad==0.92.5
pyparsing==2.0.3
pyserial==2.7
python-dateutil==2.4.2
pytz==2015.4
pywin32==219
scipy==0.15.1
six==1.9.0
zope.interface==4.1.2

[mcdermott5125-1]
PyVISA==1.7
Twisted==15.2.1
enum34==1.0.4
ipython==3.2.0
matplotlib==1.4.3
numpy==1.9.1
pylabrad==0.92.5
pyparsing==2.0.2
pyreadline==2.0
pyserial==2.7
python-dateutil==2.4.2
pytz==2015.4
pywin32==219
scipy==0.14.0
six==1.9.0
zope.interface==4.1.2

[mcdermott5125-2]
PyVISA==1.7
Twisted==15.2.1
enum34==1.0.4
ipython==3.1.0
matplotlib==1.4.3
numpy==1.9.2
pylabrad==0.92.5
pyparsing==2.0.3
pyserial==2.7
python-dateutil==2.4.1
pytz==2015.2
pywin32==219
scipy==0.15.1
six==1.9.0
zope.interface==4.1.2