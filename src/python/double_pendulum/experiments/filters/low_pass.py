import numpy as np


# Low-pass filter
def online_filter(data_measured, n, alpha):
    data_filtered = np.zeros(n)
    i = 1

    # choose an alpha value between 0 and 1, where 1 is equivalent to
    # unfiltered data
    while i < n:
        data_filtered[i] = data_measured[i] * alpha + (data_filtered[i-1] * (1.0 - alpha))
        i += 1
    return data_filtered[-1]
