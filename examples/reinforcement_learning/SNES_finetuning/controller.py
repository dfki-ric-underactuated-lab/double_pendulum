import numpy as np
import torch
from double_pendulum.controller.abstract_controller import AbstractController


class SACController(AbstractController):
    def __init__(
        self, model, dynamics_func, scaling=True, include_time=True, window_size=0, evaluating=True
    ):
        super().__init__()
        self.model = model
        self.dynamics_func = dynamics_func
        self.scaling = scaling
        self.window_size = window_size
        self.include_time = include_time
        self.old_state = [
            [0] * (5 if include_time else 4) for _ in range(self.window_size)
        ]
        self.old_action = [[0.0] for _ in range(self.window_size)]
        self.timestep = 0
        self.evaluating = evaluating

    def get_state(self, obs, time):
        if self.window_size > 0:
            return np.concatenate(
                [np.reshape(self.old_state, (-1)), np.reshape(self.old_action, (-1))]
            )
        else:
            if self.include_time:
                return list(obs) + [time / 10]
            else:
                # add gaussian noise to velocities
                # obs[2:] = obs[2:] + np.random.randn(2) * 0.01
                return obs

    def update_old_state(self, obs, t):
        if self.include_time:
            self.old_state = self.old_state[1:] + [list(obs) + [t / 10]]
        else:
            self.old_state = self.old_state[1:] + [list(obs)]

    def update_old_action(self, action):
        self.old_action = self.old_action[1:] + [action[0]]

    def get_control_output_(self, x, t=None):
        self.timestep += 1
        obs = self.dynamics_func.normalize_state(x)
        self.update_old_state(obs, t)

        action = self.model.predict(self.get_state(obs, t), deterministic=True)

        # add gaussian noise to action
        if not self.evaluating:
            a = action[0][0]
            a = torch.atanh(torch.tensor(a))
            a += np.random.randn() * 0.1
            a = torch.tanh(a).numpy()
            action[0][0] = np.clip(a, a_min=-1, a_max=+1)

        self.update_old_action(action)
        return self.dynamics_func.unscale_action(action)


def load_controller(dynamics_func, model, window_size, include_time, evaluating):
    name = "sac"
    leaderboard_config = {
        "csv_path": name + "/sim_swingup.csv",
        "name": name,
        "simple_name": "sac",
        "short_description": "SAC for both swingup and stabilisation",
        "readme_path": f"readmes/{name}.md",
        "username": "MarcoCali0",
    }
    controller = SACController(
        model=model,
        dynamics_func=dynamics_func,
        window_size=window_size,
        include_time=include_time,
        evaluating=evaluating
    )
    controller.init()
    return controller, leaderboard_config
