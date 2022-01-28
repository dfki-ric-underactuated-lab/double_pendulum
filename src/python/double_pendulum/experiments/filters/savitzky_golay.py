from scipy.signal import savgol_filter
import pandas as pd


def savitzky_golay_filter(data_measured, window, degree):
    """
    Savitzky-Golay Filter
    Local least-squares polynomial approximation.
    Delay = (window-1)/2 * delta_t
    """
    data_filtered = pd.DataFrame(columns=data_measured.columns)
    data_filtered['time'] = data_measured['time']
    data_filtered[['pos', 'vel', 'torque']] = savgol_filter(
        data_measured[['pos', 'vel', 'torque']],
        window,
        degree,
        mode='nearest',
        axis=0)
    return data_filtered
