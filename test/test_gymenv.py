"""
Unit Tests
==========
"""

import unittest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import (
    CustomEnv,
    double_pendulum_dynamics_func,
)


def reward_func(observation, action):
    return -(
        observation[0] ** 2.0
        + (observation[1] + 1.0) * (observation[1] - 1.0)
        + observation[2] ** 2.0
        + observation[3] ** 2.0
        + 0.01 * action[0] ** 2.0
    )


def terminated_func(observation):
    return False


def noisy_reset_func():
    rand = np.random.rand(4) * 0.01
    rand[2:] = rand[2:] - 0.05
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return np.float32(observation)


class Test(unittest.TestCase):
    mpar = model_parameters()
    plant = DoublePendulumPlant(model_pars=mpar)
    simulator = Simulator(plant=plant)

    # learning environment parameters
    state_representation = 2
    obs_space = obs_space = gym.spaces.Box(
        np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
    max_steps = 100
    robot = "acrobot"
    dt = 0.002
    integrator = "runge_kutta"

    dynamics_func = double_pendulum_dynamics_func(
        simulator=simulator,
        dt=dt,
        integrator=integrator,
        robot=robot,
        state_representation=state_representation,
    )

    env = CustomEnv(
        dynamics_func=dynamics_func,
        reward_func=reward_func,
        terminated_func=terminated_func,
        reset_func=noisy_reset_func,
        obs_space=obs_space,
        act_space=act_space,
        max_episode_steps=100,
    )

    def test_0_sb3_checker(self):
        sb3_check_env(self.env)

    def test_1_sb3_checker(self):
        gymnasium_check_env(self.env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
