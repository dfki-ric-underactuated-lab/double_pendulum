import os
import yaml
import numpy as np

from double_pendulum.filter.abstract_filter import AbstractFilter


class identity_filter(AbstractFilter):
    def __init__(
        self,
        filt_velocity_cut=-1.0,
    ):
        super().__init__()

        self.filt_velocity_cut = filt_velocity_cut

    def init_(self):
        pass

    def save_(self, save_dir):
        par_dict = {
            "filt_velocity_cut": self.filt_velocity_cut,
        }
        with open(os.path.join(save_dir, "filter_identity_parameters.yml"), "w") as f:
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

        return x_filt
