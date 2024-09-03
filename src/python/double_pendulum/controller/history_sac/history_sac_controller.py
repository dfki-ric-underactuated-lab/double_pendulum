from double_pendulum.controller.abstract_controller import AbstractController
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

from double_pendulum.controller.history_sac.utils import general_dynamics
from double_pendulum.controller.history_sac.history_sac import HistorySAC, HistoryEnv


class HistorySACController(AbstractController):
    def __init__(self, env_type, model_path):
        super().__init__()

        dt = 0.02
        max_torque = 6
        dynamics_func = general_dynamics(env_type, dt, max_torque)

        envs = make_vec_env(
            env_id=HistoryEnv,
            n_envs=1,
            env_kwargs={
                "dynamic_function": dynamics_func,
                "reward_function": None,
                "termination_function": None,
                "reset_function": lambda: [0, 0, 0, 0],
                "torque_limit": max_torque
            },
            vec_env_cls=DummyVecEnv
        )

        self.model = HistorySAC.load(
            model_path,
            env=envs,
            print_system_info=True,
            env_type=env_type
        )

        self.precision = 10000
        self.lowpass = 0.0
        self.dynamics_func = dynamics_func
        self.simulator = self.dynamics_func.simulator
        self.dt = self.dynamics_func.dt
        self.scaling = dynamics_func.scaling
        self.integrator = dynamics_func.integrator
        self.controller_dt = np.rint(self.dt * self.precision).astype(int)
        self.history = None
        self.last_action = None
        self.n = None
        self.last_u = None
        self.reset()

    def reset(self):
        super().reset()
        self.history = {'X': [], 'U': []}
        self.last_action = 0.0
        self.n = 1
        self.last_u = None
        self.model.env.envs[0].env.reset()

    def get_control_output_(self, x, t=None):
        if self.n == 1 and t > 0:
            self.n = np.rint(self.dt / np.round(t, decimals=5)).astype(int)

        env = self.model.env.envs[0].env
        obs = self.dynamics_func.normalize_state(x)
        rounded_t = np.rint(t * self.precision).astype(int)
        if rounded_t % self.controller_dt == 0 and t > 0.0:
            env.history['T'].append(np.round(t, decimals=5))
        self.history['X'].append(obs)
        self.history['U'].append(self.last_action)

        env.history['U_con'] = self.history['U'][::-1][::self.n][::-1].copy()
        env.history['X_meas'] = self.history['X'][::-1][::self.n][::-1].copy()
        action, _ = self.model.predict(observation=obs.reshape(1, -1), deterministic=True)
        new_action = self.lowpass * self.last_action + (1 - self.lowpass) * action.item()
        if self.last_action == 0.0:
            new_action = action.item()
        self.last_action = new_action
        return self.dynamics_func.unscale_action(new_action)
