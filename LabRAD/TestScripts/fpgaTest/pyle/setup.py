"""pyle of standard datataking code
"""

classifications = """\
Development Status :: 4 - Beta
Environment :: Console
Environment :: Web Environment
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering"""

from distutils.core import setup

doclines = __doc__.split('\n')

setup(
    name = 'pyle',
    version = '10.03',
    author = 'Martinis Group',
    #author_email = 'pyle@physics.ucsb.edu',
    license = 'http://www.gnu.org/licenses/gpl-2.0.html',
    platforms = ['ANY'],
    
    url = 'http://www.physics.ucsb.edu/~martinisgroup',
    download_url = '',
    
    description = doclines[0],
    long_description = '\n'.join(doclines[2:]),
    classifiers = classifications.split('\n'),
    
    requires = ['pylabrad'],
    provides = ['pyle'],
    packages = [
        'pyle',
        'pyle.dataking',
        'pyle.util',
        ],
    package_data = {
        'pyle': ['LICENSE.txt'],
        },
    )
