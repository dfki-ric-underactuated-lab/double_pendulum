import pathlib
import json
import jax
import jax.numpy as jnp
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.DQN.networks import DQN
from double_pendulum.controller.DQN.utils import load_pickled_data


class DQNController(AbstractController):
    def __init__(self, experiment_path, actions, dynamics_func, dt):
        super().__init__()
        p = json.load(open(experiment_path + "parameters.json"))  # p for parameters

        self.actions = actions
        self.q = DQN((4,), p["n_actions"], p["gamma"], p["layers"], jax.random.PRNGKey(0), None, None, None)
        self.q.params = load_pickled_data(list(pathlib.Path(experiment_path).glob("*best_online_params"))[0])
        self.dynamics_func = dynamics_func
        self.dt = dt

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action = self.actions[self.q.best_action(self.q.params, obs, jax.random.PRNGKey(0))]
        u = self.dynamics_func.unscale_action(jnp.array([action]))
        return u
