import numpy as np
import warnings
    
def mean_time(t, min_threshold=0, max_threshold=1253):
    """
    Take a switch probability result array from the PreAmp timer, and 
    compute mean switching time using the specified thresholds. Timing
    data is assumed to be a numpy array.
    """
    t = t[np.logical_and(t > min_threshold, t < max_threshold)]
    if np.size(t) > 0:
        t_mean = np.mean(t)
        t_std = np.std(t)
    else:
        t_mean = np.nan
        t_std = np.nan

    return t_mean, t_std

def mean_time_diff(t, min_threshold=0, max_threshold=1253):
    """
    Take a switch probability result array from the PreAmp timers, and
    compute mean switching time using the specified thresholds.
    """
    dt = t[0][:] - t[1][:]
    t0_mask = np.logical_and(t[0,:] > min_threshold, t[0,:] < max_threshold)
    t1_mask = np.logical_and(t[1,:] > min_threshold, t[1,:] < max_threshold)
    dt = dt[np.logical_and(t0_mask, t1_mask)]
    if np.size(dt) > 0:
        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
    else:
        dt_mean = np.nan
        dt_std = np.nan

    return dt_mean, dt_std

def prob(t, min_threshold=0, max_threshold=1253):
    """
    Take a switch probability result array from the PreAmp timer, and
    compute switching probability using the specified thresholds.
    """
    return float(np.size(t[np.logical_and(t > min_threshold, t < max_threshold)])) / float(np.size(t))
    
def outcomes(t, min_threshold=0, max_threshold=1253):
    """
    Take a switch probability result array from the PreAmp timer, and
    convert to a numpy array of 0 or 1 based on the thresholds.
    """
    def _threshold(x):
        if x > min_threshold and x < max_threshold:
            return 1
        else:
            return 0
    
    threshold_vectorized = np.vectorize(_threshold)

    return threshold_vectorized(t)
    
def corr_coef_from_outcomes(outcomes):
    """
    Compute correrlation coefficient from an array of switching
    outcomes.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.corrcoef(outcomes[0,:], outcomes[1,:])[0,1]

def software_demod(t, freq, Is, Qs):
    """
    Demodulate I and Q data in software. This method uses
    ADC frequency for demodulation. 
    
    Input:
        t: time vector during which to demodulate data (ns).
        freq: demodulation frequency (GHz).
        Is: I data.
        Qs: Q data.
    Output:
        Id, Qd: demodulated I and Q.
    """
    demod = 2 * np.pi * t * freq
    
    Sv = np.sin(demod)
    Cv = np.cos(demod)

    Id = np.mean(Is * Cv - Qs * Sv)
    Qd = np.mean(Is * Sv + Qs * Cv)
    
    return Id, Qd