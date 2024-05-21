import os
import yaml
import numpy as np

from double_pendulum.filter.abstract_filter import AbstractFilter


class lowpass_filter(AbstractFilter):
    def __init__(
        self,
        alpha=[1.0, 1.0, 0.3, 0.3],
        x0=[0.0, 0.0, 0.0, 0.0],
        filt_velocity_cut=-1.0,
    ):
        super().__init__()

        self.alpha = np.asarray(alpha)
        self.x0 = x0
        self.filt_velocity_cut = filt_velocity_cut

        self.init_()

    def init_(self):
        self.x_hist = [self.x0]
        self.x_filt_hist = [self.x0]

    def save_(self, save_dir):
        par_dict = {
            "alpha": list(self.alpha),
            "x0": list(self.x0),
            "filt_velocity_cut": self.filt_velocity_cut,
        }
        with open(os.path.join(save_dir, "filter_lowpass_parameters.yml"), "w") as f:
            yaml.dump(par_dict, f)

    def get_filtered_state_(self, x, u, t=None):
        x_filt = np.copy(x)

        # print("x=", x)
        # print("u=", u)
        # velocity cut
        if self.filt_velocity_cut > 0.0:
            x_filt[2] = np.where(
                np.abs(x_filt[2]) < self.filt_velocity_cut, 0, x_filt[2]
            )
            x_filt[3] = np.where(
                np.abs(x_filt[3]) < self.filt_velocity_cut, 0, x_filt[3]
            )

        x_filt = (1.0 - self.alpha) * self.x_filt_hist[-1] + self.alpha * x_filt
        # print("x_filt=", x_filt)
        return x_filt
