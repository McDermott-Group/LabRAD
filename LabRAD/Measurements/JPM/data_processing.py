import numpy as np
    
def mean_time_from_array(self, t, threshold):
    """
    Take a switch probability result array from the PreAmp timer, and compute
    mean switching time using the specified threshold.
    """
    t = np.array(t)
    t = t[t < threshold]
    if np.size(t) > 0:
        t_mean = np.mean(t)
        t_std = np.std(t)
    else:
        t_mean = -1
        t_std = 0

    return t_mean, t_std

def mean_time_diff_from_array(self, t, threshold):
    """
    Take a switch probability result array from the PreAmp timers, and compute
    mean switching time using the specified threshold.
    """
    t = np.array(t)
    dt = t[0][:] - t[1][:]
    dt = dt[np.logical_and(t[0][:] < threshold, t[1][:] < threshold)]
    if np.size(dt) > 0:
        dt_mean = np.mean(dt)
        dt_std = np.std(dt)
    else:
        dt_mean = 0
        dt_std = 0

    return dt_mean, dt_std

def prob_from_array(self, t, threshold):
    """
    Take a switch probability result array from the PreAmp timer, and compute
    switching probability using the specified threshold.
    """
    t = np.array(t)

    return float(np.size(t[t < threshold])) / float(np.size(t))
    
def preamp_counts_to_array(self, t, threshold):
    """
    Take a switch probability result array from the PreAmp timer, and convert
    to a numpy array of 0 or 1 based on the threshold.
    """
    def _Threshold(x):
        if x < threshold:
            return 1
        else:
            return 0
    
    ThresholdVectorized = np.vectorize(_Threshold)

    return ThresholdVectorized(t)