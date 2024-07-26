from pathlib import Path

import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.AR_EAPO.ar_eapo import AR_EAPO


class AR_EAPOController(AbstractController):
    def __init__(
        self,
        model_path: str | Path,
        robot: str,
        max_torque=6.0,
        max_velocity=20.0,
        deterministic: bool = True,
    ):
        super().__init__()

        self.model: AR_EAPO = AR_EAPO.load(model_path)
        self.torque_limits = np.array(
            [max_torque * (robot == "pendubot"), max_torque * (robot == "acrobot")],
            np.float64,
        )
        self.max_velocity = max_velocity
        self.vec_normalize = self.model.get_vec_normalize_env()
        self.deterministic = deterministic

    def get_control_output_(self, x, t=None):
        obs = self._scale_obs(x)

        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)

        action, _ = self.model.predict(obs, deterministic=self.deterministic)

        return self.torque_limits * action

    def _scale_obs(self, state: np.ndarray):
        observation = np.array(
            [
                (state[0] % (2 * np.pi) - np.pi) / np.pi,
                (state[1] % (2 * np.pi) - np.pi) / np.pi,
                np.clip(state[2], -self.max_velocity, self.max_velocity)
                / self.max_velocity,
                np.clip(state[3], -self.max_velocity, self.max_velocity)
                / self.max_velocity,
            ]
        )
        return observation
