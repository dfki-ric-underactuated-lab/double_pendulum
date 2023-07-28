import copy
import signal
from scipy import signal as scipy_signal
import numpy as np


# Low-pass filter
def lowpass_filter(data_measured, alpha):
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


class lowpass_filter_rt():
    def __init__(self,
                 dim_x=4,
                 alpha=[1., 1., 0.3, 0.3],
                 x0=[0., 0., 0., 0.]):
        self.alpha = np.asarray(alpha)
        # self.dim_x = dim_x
        self.data = np.asarray(x0).reshape(1, len(x0))
        self.data = [x0]

    def __call__(self, x, u=None):
        x_est = (1.-self.alpha)*self.data[-1] + self.alpha*x
        # self.data = np.append(self.data, [np.copy(x_est)], axis=0)
        self.data.append(x_est)
        # print(self.data[-2], x, x_est, np.shape(self.data), self.data)
        return np.copy(x_est)


class butter_filter_rt():
    def __init__(self,
                 dof=2, cutoff=0.5, dt=0.002,
                 x0=[0., 0., 0., 0.]):
        self.dof = dof
        # self.dim_x = dim_x
        self.data = [np.array(x0)]
        self.data_filt = [np.array(x0)]
        self.cutoff = cutoff
        self.b, self.a = scipy_signal.butter(1, self.cutoff)
        self.dt = dt

    def __call__(self, x, u=None):
        pos = x[:self.dof]

        # numeric diff
        vel = (pos - self.data[-1][:self.dof]) / self.dt

        # filtering
        vel = (self.b[0] * vel + self.b[1] * self.data[-1][self.dof:] - self.a[1] * self.data_filt[-1][self.dof:]) / self.a[0]

        self.data.append(x)

        x_ = copy.deepcopy(x)
        x_[self.dof:] = vel
        self.data_filt.append(x_)

        return np.array(self.data_filt[-1])
