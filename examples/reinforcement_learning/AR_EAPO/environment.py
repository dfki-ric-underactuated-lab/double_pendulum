from pathlib import Path
from typing import TypeAlias, Literal, Any

import numpy as np
from numpy.typing import ArrayLike
from gymnasium import Env, spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func


ROBOTS: TypeAlias = Literal["acrobot", "pendubot"]


GOAL = np.array([np.pi, 0.0, 0.0, 0.0], dtype=np.float64)


class Environment(Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        dynamics_func: double_pendulum_dynamics_func,
        scaling: bool,
        max_steps: int,
        Q: ArrayLike,
        R: float,
        reward_scale: float,
        random_truncation_probability: float,
    ) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float64)

        self._dynamics_func = dynamics_func
        self.scaling = scaling
        self.max_steps = max_steps

        # Parameters for the reward function
        self.Q = np.diag(Q)
        self.R = np.array([[R]])
        self.reward_scale = reward_scale

        self.p_trunc = random_truncation_probability

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._num_steps = 0

        rand = self.np_random.standard_normal(4, np.float64) * 0.01
        self.x = rand

        if self.scaling:
            observation = self._dynamics_func.normalize_state(rand)
        else:
            observation = rand.copy()

        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.x = self._dynamics_func(self.x, action, False)
        reward = self._reward(self.x, action)

        # Truncation
        truncated = False
        self._num_steps += 1
        if self._num_steps >= self.max_steps:
            truncated = True
        elif self.np_random.random() < self.p_trunc:
            truncated = True

        if self.scaling:
            observation = self._dynamics_func.normalize_state(self.x)
        else:
            observation = self.x.copy()

        # Termination is always False.
        return observation, reward, False, truncated, {}

    def _reward(self, x: np.ndarray, action: np.ndarray) -> float:
        x = self._wrap_angles(x)
        u = self._dynamics_func.unscale_action(action)
        diff = x - GOAL
        r = -np.einsum("i, ij, j", diff, self.Q, diff)
        r -= np.einsum("i, ij, j", u, self.R, u)
        r *= self.reward_scale
        return r

    def _wrap_angles(self, x: np.ndarray):
        x = x.copy()
        x[0] = x[0] % (2 * np.pi)  # [0, 2π]
        x[1] = (x[1] + np.pi) % (2 * np.pi) - np.pi  # [-π, π]
        return x


def make_env(
    robot: ROBOTS,
    model_params_path: str | Path,
    dt: float,
    max_torque: float,
    max_velocity: float,
    scaling: bool,
    max_steps: int,
    Q: ArrayLike,
    R: float,
    reward_scale: float,
    random_truncation_probability: float,
    n_envs: int,
    norm_obs: bool,
    seed: int | None = None,
    training: bool = True,
) -> tuple[VecEnv, double_pendulum_dynamics_func]:
    torque_limits = np.array(
        [max_torque * (robot == "pendubot"), max_torque * (robot == "acrobot")],
        np.float64,
    )

    model_params = model_parameters(filepath=str(model_params_path))
    model_params.set_motor_inertia(0.0)
    model_params.set_damping([0.0, 0.0])
    model_params.set_cfric([0.0, 0.0])
    model_params.set_torque_limit(torque_limits)

    plant = SymbolicDoublePendulum(model_pars=model_params)
    simulator = Simulator(plant)
    dynamics_func = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        robot=robot,
        scaling=False,  # We do scaling in the environment
        torque_limit=torque_limits,
        max_velocity=max_velocity,
    )

    env_kwargs = dict(
        dynamics_func=dynamics_func,
        scaling=scaling,
        max_steps=max_steps,
        Q=Q,
        R=R,
        reward_scale=reward_scale,
        random_truncation_probability=random_truncation_probability,
    )

    env = make_vec_env(Environment, n_envs, seed, env_kwargs=env_kwargs)
    if norm_obs:
        env = VecNormalize(
            env, training=training, norm_obs=True, norm_reward=False, gamma=1.0
        )

    return env, dynamics_func
