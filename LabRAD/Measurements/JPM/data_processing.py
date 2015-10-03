import numpy as np
    
def mean_time_from_array(t, threshold):
    """
    Take a switch probability result array from the PreAmp timer, and 
    compute mean switching time using the specified threshold.
    """
    t = np.array(t)
    t = t[t < threshold]
    if np.size(t) > 0:
        t_mean = np.mean(t)
        t_std = np.std(t)
    else:
        t_mean = np.nan
        t_std = np.nan

    return t_mean, t_std

def mean_time_diff_from_array(t, threshold):
    """
    Take a switch probability result array from the PreAmp timers, and
    compute mean switching time using the specified threshold.
    """
    t = np.array(t)
    dt = t[0][:] - t[1][:]
    dt = dt[np.logical_and(t[0][:] < threshold, t[1][:] < threshold)]
    if np.size(dt) > 0:
        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
    else:
        dt_mean = np.nan
        dt_std = np.nan

    return dt_mean, dt_std

def prob_from_array(t, threshold):
    """
    Take a switch probability result array from the PreAmp timer, and
    compute switching probability using the specified threshold.
    """
    t = np.array(t)

    return float(np.size(t[t < threshold])) / float(np.size(t))
    
def outcomes_from_array(t, threshold):
    """
    Take a switch probability result array from the PreAmp timer, and
    convert to a numpy array of 0 or 1 based on the threshold.
    """
    def _threshold(x):
        if x < threshold:
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
    return np.corrcoef(outcomes[0, :], outcomes[1, :])[0, 1]

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