#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import multiprocessing as mp

import labrad.units as units


class TimeoutError(Exception):
    """Timeout error."""
    pass


def _wrap(q, func, args, **kwargs):
    """Put the execution result into the multiprocessing queue."""
    q.put(func(*args, **kwargs))
    
def timeout(timeout, func, *args, **kwargs):
    """
    Timed-out execution of a function in a separate process.
    
    Inputs:
        timeout: time in seconds.
        func: specific function to run.
        *args and **kwargs: input argumens of the function.
    Output:
        result: result of the function call, otherwise, an exception is
        raisen.
    """
    if isinstance(timeout, units.Value):
        timeout = timeout['s']
        
    q = mp.Queue()
    p = mp.Process(target=_wrap, args=(q, func, args), kwargs=kwargs)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError('Execution of ' + str(func) + 
                ' timed-out after ' + str(timeout) + ' sec.')
    else:
        return q.get()