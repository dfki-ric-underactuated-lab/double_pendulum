import os
import yaml
import numpy as np
from scipy import signal

from double_pendulum.filter.abstract_filter import AbstractFilter


class butterworth_filter(AbstractFilter):
    """
    creates a 3. order Butterworth lowpass filter with a cutoff of 0.2
    times the Nyquist frequency or 200 Hz, returning enumerator (b) and
    denominator (a) polynomials for a Infinite Impulse Response (IIR)
    filter
    """

    def __init__(
        self,
        cutoff=0.5,
        dt=0.002,
        x0=[0.0, 0.0, 0.0, 0.0],
        filt_velocity_cut=-1.0,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.dt = dt
        self.x0 = x0
        self.filt_velocity_cut = filt_velocity_cut
        self.b, self.a = signal.butter(1, self.cutoff)

        self.init_()

    def init_(self):
        self.x_hist = [self.x0]
        self.x_filt_hist = [self.x0]

    def save_(self, save_dir):
        par_dict = {
            "cutoff": self.cutoff,
            "dt": self.dt,
            "x0": list(self.x0),
            "a": self.a,
            "b": self.b,
            "filt_velocity_cut": self.filt_velocity_cut,
        }
        with open(
            os.path.join(save_dir, "filter_butterworth_parameters.yml"), "w"
        ) as f:
            yaml.dump(par_dict, f)

    def get_filtered_state_(self, x, u, t=None):
        x_filt = np.copy(x)

        # velocity cut
        if self.filt_velocity_cut > 0.0:
            x_filt[2] = np.where(
                np.abs(x_filt[2]) < self.filt_velocity_cut, 0, x_filt[2]
            )
            x_filt[3] = np.where(
                np.abs(x_filt[3]) < self.filt_velocity_cut, 0, x_filt[3]
            )

        vel_filt = x_filt[2:]

        vel_filt = (
            self.b[0] * vel_filt
            + self.b[1] * self.x_hist[-1][2:]
            - self.a[1] * self.x_filt_hist[-1][2:]
        ) / self.a[0]

        x_filt[2:] = vel_filt

        return x_filt
