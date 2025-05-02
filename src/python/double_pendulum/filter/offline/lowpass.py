import numpy as np


# Low-pass filter
def lowpass_filter_offline(data_measured, alpha):
    """
    choose an alpha value between 0 and 1, where 1 is equivalent to
    unfiltered data
    """
    N = len(data_measured)
    data_filtered = np.zeros(N)
    data_filtered[0] = data_measured[0]

    for i in range(1, N):
        data_filtered[i] = alpha*data_measured[i] + \
                           (1.-alpha)*data_filtered[i-1]
    return data_filtered
