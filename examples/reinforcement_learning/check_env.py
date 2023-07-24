import os
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.env_util import make_vec_env

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import (
    CustomEnv,
    double_pendulum_dynamics_func,
)

# model parameters
design = "design_A.0"
model = "model_2.0"
robot = "acrobot"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)
dt = 0.002
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 2
obs_space = obs_space = gym.spaces.Box(
    np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
)
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
max_steps = 100


dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
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


env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=100,
)

# stable baselines check
sb3_check_env(env)
print("StableBaselines3 env_check successful.")

# gymnasium env check
gymnasium_check_env(env, skip_render_check=True)
print("Gymnasium env_check successful.")

# # vectorized environment
# envs = make_vec_env(
#     env_id=CustomEnv,
#     n_envs=3,
#     env_kwargs={
#         "dynamics_func": dynamics_func,
#         "reward_func": reward_func,
#         "terminated_func": terminated_func,
#         "reset_func": noisy_reset_func,
#         "obs_space": obs_space,
#         "act_space": act_space,
#         "max_episode_steps": max_steps,
#     },
# )
