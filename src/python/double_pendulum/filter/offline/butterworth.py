from scipy import signal


# Butterworth filter
def butterworth_filter_offline(data_measured, order, cutoff):
    """
    creates a 3. order Butterworth lowpass filter with a cutoff of 0.2 times
    the Nyquist frequency or 200 Hz, returning enumerator (b) and
    denominator (a) polynomials for a Infinite Impulse Response (IIR) filter

    The result should be approximately xlow, with no phase shift.
    >>> b, a = signal.butter(8, 0.125)
    >>> y = signal.filtfilt(b, a, x, padlen=150)
    >>> np.abs(y - xlow).max()
    9.1086182074789912e-06
    """
    b, a = signal.butter(order, cutoff)

    # applies a linear digital filter twice, once forward and once backwards.
    # The combined filter has zero phase and
    # a filter order twice that of the original.
    data_filtered = signal.filtfilt(b, a, data_measured)
    return data_filtered
