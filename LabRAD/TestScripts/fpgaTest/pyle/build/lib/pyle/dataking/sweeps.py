import numpy as np

import pyle
from pyle.pipeline import pmap, returnValue
from pyle.util import getch


def prepDataset(sample, name, axes=None, dependents=None, measure=None, cxn=None, kw=None, explicitIndependents=None):
    """Prepare dataset for a sweep.
    
    This function builds a dictionary of keyword arguments to be used to create
    a Dataset object for a sweep.  Sample should be a dict-like object
    (usually a copy of the sample as returned by loadQubits) that contains current
    parameter settings.  Name is the name of the dataset, which will get prepended
    with a list indicating which qubits are in the sample config, and which of them
    are to be measured.  kw is a dictionary of additional parameters that should
    be added to the dataset.
    
    Axes can be specified explicitly as a tuple of (<name>, <unit>), or else
    by value for use with grid sweeps.  In the latter case (grid sweep), you should
    specify axes as (<value>, <name>).  If value is iterable, the axis will be
    added to the list of dependent variables so the value can be swept (we look
    at element [0] to get the units); if value is not iterable it will be added
    to the dictionary of static parameters.
    
    Dependents is either a list of (<name>, <label>, <unit>) designations for the
    dependent variables, or None.  If no list is provided, then the dependents are
    assumed to be probabilities.  In this case, the measure variable is used to
    determine the appropriate set of probabilities: for one qubit, we assume only
    P1 will be measured, while for N qubits all 2^N probabilities are assumed to
    be measured, in the order P00...00, P00...01, P00...10,..., P11...11.  If this
    is not what you want, you must specify the independent variables explicitly.
    
    Note that measure can be None (all qubits assumed to be measured), an integer
    (just one qubit measured, identified by index in sample['config']) or a list
    of integers (multiple qubits measured).
    
    explicitIndependents are independent variables which are explicitly included
    in the argument of returnValue in func. These variables are added after any
    independent variables given by axes. It must be either None (default) or a
    list of tuples. If explicitIndependents is not None, the tuples must be of the
    form ('label','unit'), where 'label' is the label for the variable, which has
    unit 'unit'.
    """
    conf = list(sample['config'])
    
    # copy parameters
    kw = {} if kw is None else dict(kw)
    kw.update(sample) # copy all sample data
    
    if measure is None:
        measure = range(len(conf))
    elif isinstance(measure, (int, long)):
        measure = [measure]
    
    if hasattr(measure, 'params'):
        # this is a Measurer
        kw.update(measure.params())
    else:
        kw['measure'] = measure
    
    # update dataset name to reflect which qubits are measured
    for i, q in enumerate(conf):
        if i in kw['measure']:
            conf[i] = '|%s>' % q
    name = '%s: %s' % (', '.join(conf), name)
    
    # create list of independent vars
    independents = []
    for param, label in axes:
        if isinstance(param, str):
            # param specified as string name
            independents.append((param, label))
        elif np.iterable(param):
            # param value will be swept
            try:
                units = param[0].units
            except Exception:
                units = ''
            independents.append((label, units))
        else:
            # param value is static
            kw[label] = param
    if explicitIndependents is not None:
        for indep in explicitIndependents:
            independents.append(indep)
    
    # create list of dependent vars
    if dependents is None:
        if hasattr(measure, 'dependents'):
            # this is a Measurer
            dependents = measure.dependents()
        else:
            n = len(measure)
            if n == 1:
                labels = ['|1>']
            else:
                labels = ['|%s>' % bin(i)[2:].rjust(n,'0') for i in xrange(2**n)]
            dependents = [('Probability', s, '') for s in labels]
    
    return pyle.Dataset(
        path=list(sample._dir),
        name=name,
        independents=independents,
        dependents=dependents,
        params=kw,
        cxn=cxn
    )


def run(func, sweep, save=True, dataset=None,
        abortable=True, abortPrefix=[],
        collect=True, noisy=True, pipesize=10):
    """Run a function pipelined over an iterable sweep.
    
    func: function that will be called once for each value in the sweep.
          should be written to work with pipelining, and should
          return sequence objects ready to be sent to the qubit sequencer.
    sweep: an iterable which returns successive values to be
           passed to func
    
    abortable: if True, check for keypresses to allow the sweep to be aborted cleanly
    save: if True, create a new dataset (using ds_info) and save all data to it
    collect: if True, collect the data into an array and return it
    noisy: if True, print each row of data as it comes in
    
    dataset: a dataset that will be called with the iterable of data to be saved
    pipesize: the number of pipelined calls to func that should be run in parallel

    
    The following additional parameters usually only need to be modified for defining
    new types of sweep, such as grid:
    
    abortPrefix: passed along to checkAbort for abortable sweeps
    """
    with pyle.QubitSequencer() as sequencer:
        # wrap the sweep iterator to handle keypresses
        if abortable:
            sweep = checkAbort(sweep, prefix=abortPrefix)
        
        # wrap the function to pass the qubit sequencer as the first param
        def wrapped(val):
            ans = yield func(sequencer, val)
            ans = np.asarray(ans)
            if noisy:
                if len(ans.shape) == 1:
                    rows = [ans]
                else:
                    rows = ans
                for row in rows:
                    print ' '.join(('%0.3g' % v).ljust(8) for v in row)
            returnValue(ans)
        
        # Build the generator that returns results of func. Note that this generator
        # doesn't execute any code yet, and won't until .next() is called. 
        iter = pmap(wrapped, sweep, size=pipesize)
        #Massage iter so that the datavault can catch incoming data
        if save and dataset:
            iter = dataset.capture(iter)
        
        # run the iterable, and either collect or discard
        if collect:
            return np.vstack(iter)
        else:
            for _ in iter: pass

def runSim(func, sweep, save=True, dataset=None,
        abortable=True, abortPrefix=[],
        collect=True, noisy=True, pipesize=10):
    """Run a function pipelined over an iterable sweep.
    
    func: function that will be called once for each value in the sweep.
          should be written to work with pipelining, and should
          return sequence objects ready to be sent to the qubit sequencer.
    sweep: an iterable which returns successive values to be
           passed to func
    
    abortable: if True, check for keypresses to allow the sweep to be aborted cleanly
    save: if True, create a new dataset (using ds_info) and save all data to it
    collect: if True, collect the data into an array and return it
    noisy: if True, print each row of data as it comes in
    
    dataset: a dataset that will be called with the iterable of data to be saved
    pipesize: the number of pipelined calls to func that should be run in parallel

    
    The following additional parameters usually only need to be modified for defining
    new types of sweep, such as grid:
    
    abortPrefix: passed along to checkAbort for abortable sweeps
    """
    # wrap the sweep iterator to handle keypresses
    if abortable:
        sweep = checkAbort(sweep, prefix=abortPrefix)
    
    # wrap the function to pass the qubit sequencer as the first param
    def wrapped(val):
        ans = yield func(val)
        ans = np.asarray(ans)
        if noisy:
            if len(ans.shape) == 1:
                rows = [ans]
            else:
                rows = ans
            for row in rows:
                print ' '.join(('%0.3g' % v).ljust(8) for v in row)
        returnValue(ans)
    
    # Build the generator that returns results of func. Note that this generator
    # doesn't execute any code yet, and won't until .next() is called. 
    iter = pmap(wrapped, sweep, size=pipesize)
    #Massage iter so that the datavault can catch incoming data
    if save and dataset:
        iter = dataset.capture(iter)
    
    # run the iterable, and either collect or discard
    if collect:
        return np.vstack(iter)
    else:
        for _ in iter: pass
            
def grid(func, axes, **kw):
    """Run a pipelined sweep on a grid over the given list of axes.
    
    The axes should be specified as a list of (value, label) tuples.
    We iterate over each axis that is iterable, leaving others constant.
    Func should be written to return only the dependent variable data
    (e.g. probabilities), and the independent variables that are being
    swept will be prepended automatically before the data is passed along.
    
    All other keyword arguments to this function are passed directly to run.
    """
    def gridSweep(axes):
        if not len(axes):
            yield (), ()
        else:
            (param, _label), rest = axes[0], axes[1:]
            if np.iterable(param): # TODO: different way to detect if something should be swept
                for val in param:
                    for all, swept in gridSweep(rest):
                        yield (val,) + all, (val,) + swept
            else:
                for all, swept in gridSweep(rest):
                    yield (param,) + all, swept
    
    # pass in all params to the function, but only prepend swept params to data
    def wrapped(server, args):
        all, swept = args
        ans = yield func(server, *all)
        ans = np.asarray(ans)
        pre = np.asarray(swept)
        if len(ans.shape) != 1:
            pre = np.tile([pre], (ans.shape[0], 1))
        returnValue(np.hstack((pre, ans)))
    
    return run(wrapped, gridSweep(axes), abortPrefix=[1], **kw)
    
def gridSim(func, axes, **kw):
    """Run a pipelined sweep on a grid over the given list of axes.
    
    The axes should be specified as a list of (value, label) tuples.
    We iterate over each axis that is iterable, leaving others constant.
    Func should be written to return only the dependent variable data
    (e.g. probabilities), and the independent variables that are being
    swept will be prepended automatically before the data is passed along.
    
    All other keyword arguments to this function are passed directly to run.
    """
    def gridSweep(axes):
        if not len(axes):
            yield (), ()
        else:
            (param, _label), rest = axes[0], axes[1:]
            if np.iterable(param): # TODO: different way to detect if something should be swept
                for val in param:
                    for all, swept in gridSweep(rest):
                        yield (val,) + all, (val,) + swept
            else:
                for all, swept in gridSweep(rest):
                    yield (param,) + all, swept
    
    # pass in all params to the function, but only prepend swept params to data
    def wrapped(args):
        all, swept = args
        ans = yield func(*all)
        ans = np.asarray(ans)
        pre = np.asarray(swept)
        if len(ans.shape) != 1:
            pre = np.tile([pre], (ans.shape[0], 1))
        returnValue(np.hstack((pre, ans)))
    
    return runSim(wrapped, gridSweep(axes), abortPrefix=[1], **kw)


def checkAbort(iterable, labels=[], prefix=[]):
    """Wrap an iterator to allow it to be aborted during iteration.
    
    Pressing ESC will cause the iterable to abort immediately.
    Alternately, pressing a number key (1, 2, 3, etc.) will abort
    the next time there is a change at a specific index in the value
    produced by the iterable.  This assumes that the source iterable
    returns values at each step that are indexable (e.g. tuples) so
    that we can grab a particular element and check if it has changed.
    
    In addition, the optional prefix parameter allows to specify a part
    of the value at each step to be monitored for changes.  For example,
    grid sweeps produce two tuples, a tuple of all current values,
    and a second tuple of the current values of only the swept parameters
    (the tuple of all values is what gets passed to the sweep function,
    while the second tuple of just swept parameters is what gets passed
    to the data vault).  In this case, the prefix would be set to [1]
    so that we only check the second tuple for changes.
    """
    idx = -1
    last = None
    for val in iterable:
        curr = val
        for i in prefix:
            curr = curr[i]
        key = getch.getch()
        if key is not None:
            if key == '\x1b':
                print 'Abort scan'
                break
            elif hasattr(curr, '__len__') and key in [str(i+1) for i in xrange(len(curr))]:
                idx = int(key) - 1
                if labels:
                    print 'Abort scan on next change of %s' % labels[idx]
                else:
                    print 'Abort scan on next change at index %d' % idx
            elif key == '\r':
                if idx >= 0:
                    idx = -1
                    print 'Abort canceled'
        if (idx >= 0) and (last is not None):
            if curr[idx] != last[idx]:
                break
        yield val
        last = curr

