import os
import sys

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.wrap_angles import wrap_angles_diff

# from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from environment import CustomCustomEnv
from environment import (
    double_pendulum_dynamics_func_extended as double_pendulum_dynamics_func,
)
from magic import BruteMagicCallback, MagicCallback
from setproctitle import setproctitle
from simulator import CustomSimulator
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.sac.policies import MlpPolicy
from wrappers import *

np.random.seed(0)
torch.manual_seed(0)
torch.random.manual_seed(0)
torch.backends.cudnn.deterministic = True
stable_baselines3.common.utils.set_random_seed(0)


# initialize randomly either to neighbourhood of the goal configuration or the bottom configuration
def random_reset_func():
    # with 40% probability, start close to the goal
    if np.random.uniform() > 0.6:
        observation = [0.0, -1.0, 0.0, 0.0]
        observation[0] += np.random.rand() * 0.05  # safe bounds
        observation[1] += np.random.rand() * 0.05  # unsafe bounds
        observation[1] = (observation[1] - 1) % 2 - 1  # now safe
        observation[2] += np.random.rand() * 0.05
        observation[3] += np.random.rand() * 0.05

    else:
        rand = np.random.rand(4) * 0.03
        rand[2:] = rand[2:] - 0.05
        observation = [-1.0, -1.0, 0.0, 0.0] + rand

    return observation


def check_if_state_in_roa(S, rho, observation, max_velocity=50.0):
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * max_velocity,
            observation[3] * max_velocity,
        ]
    )

    x = wrap_angles_diff(s)
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho


class MyEnv(CustomCustomEnv):
    def reward_func(self, terminated, action):
        _, theta2, omega1, omega2 = self.dynamics_func.unscale_state(self.observation)
        costheta2 = np.cos(theta2)

        a = action[0]
        delta_action = np.abs(a - self.previous_action)
        lambda_delta = 0.05
        lambda_action = 0.02
        lambda_velocities = 0.01

        if not terminated:
            # roa_flag = self.check_roa(self.observation)
            # roa_reward = 0 if not roa_flag else 5
            if self.stabilisation_mode:
                reward = (
                    # roa_reward
                    +self.V()
                    + 2 * (1 + costheta2) ** 2
                    - self.T()
                )
            else:
                reward = (
                    self.V()  # for pendubot
                    - lambda_action * np.square(a)
                    - 5 * lambda_velocities * (omega1**2 + omega2**2)
                    - 3 * lambda_delta * delta_action
                )
        else:
            reward = -1.0
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed, options)

        ## noise

        # process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
        process_noise_sigmas = np.max(
            [[0] * 4, np.random.normal(loc=[0.01] * 4, scale=[0.01] * 4)], axis=0
        )

        # meas_noise_sigmas = [0.0, 0.0, 0.1, 0.1]
        meas_noise_sigmas = np.max(
            [[0] * 4, np.random.normal(loc=[0.01] * 4, scale=[0.01] * 4)], axis=0
        )

        delay_mode = "posvel"
        # delay = 0.05
        delay = np.max([0, np.random.normal(loc=0.05, scale=0.01)])

        # u_noise_sigmas = [0.01, 0.01]
        u_noise_sigmas = np.max(
            [[0, 0], np.random.normal(loc=[0.01] * 2, scale=[0.01] * 2)], axis=0
        )

        # u_responsiveness = 0.9
        u_responsiveness = np.min([np.random.normal(loc=0.9, scale=0.05), 1])

        simulator.set_process_noise(process_noise_sigmas=process_noise_sigmas)
        simulator.set_measurement_parameters(
            meas_noise_sigmas=meas_noise_sigmas, delay=delay, delay_mode=delay_mode
        )
        simulator.set_motor_parameters(
            u_noise_sigmas=u_noise_sigmas, u_responsiveness=u_responsiveness
        )

        perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
            10, dt, 2, 1.0, [0.05, 0.1], [0.4, 0.6]
        )
        self.dynamics_func.simulator.set_disturbances(
            perturbation_array=perturbation_array
        )
        return self.observation, {}


assert (
    len(sys.argv) >= 4
), "Please provide: [max torque] [robustness] [window_size (0 = no window)] [include_time] [robot]"

max_torque = float(sys.argv[1])
robustness = float(sys.argv[2])
WINDOW_SIZE = int(sys.argv[3])
INCLUDE_TIME = bool(int(sys.argv[4]))
robot = str(sys.argv[5])

flg_train_with_lqr = False
integrator = "runge_kutta"
dt = 0.01


FOLDER_ID = f"{os.path.basename(__file__)}-{max_torque}-{robustness}-{WINDOW_SIZE}-{int(INCLUDE_TIME)}-{dt}"
TERMINATION = False

# setting log path for the training
log_dir = f"./real_log_data_{robot}/SAC_training/{FOLDER_ID}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# we're loading the model with friction
design = "design_C.1"
model = "model_1.0"
model_par_path = (
    "../../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)

# model and reward parameter
max_velocity = 50
torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]

mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = CustomSimulator(
    plant=plant, robustness=robustness, max_torque=max_torque, robot=robot, model=model
)


eval_simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 2

obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
max_steps = 10 / dt

###############################################################################

n_envs = 50
training_steps = 300_000_000_000
verbose = 1
eval_freq = 10_000
n_eval_episodes = 10
# a patto che i reward istantanei siano piccoli
# 0.01 -> 1500000 -> 7
# 0.003 -> 1500000 -> 46
# 0.001 -> 1500000 -> 38
# 0.0003 -> 1500000 -> 19
learning_rate = 0.001

###############################################################################

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

eval_dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)


def zero_reset_func():
    observation = [-1.0, -1.0, 0.0, 0.0]
    return observation


###############################################################################


def wrap(env):
    if INCLUDE_TIME:
        env = TimeAwareWrapper(env, dt=dt)
    if WINDOW_SIZE > 0:
        env = StateActionHistoryWrapper(env, history_length=WINDOW_SIZE)
    return env


envs = make_vec_env(
    env_id=MyEnv,
    wrapper_class=wrap,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reset_func": random_reset_func,
        "terminates": TERMINATION,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
    },
)

eval_env = wrap(
    MyEnv(
        dynamics_func=eval_dynamics_func,
        reset_func=zero_reset_func,
        obs_space=obs_space,
        act_space=act_space,
        max_episode_steps=max_steps,
        terminates=TERMINATION,
    )
)

eval_callback = EvalCallback(
    eval_env,
    callback_after_eval=BruteMagicCallback(
        f"./real_models_{robot}/",
        folder_id=FOLDER_ID,
        dynamics_func=eval_dynamics_func,
        robot=robot,
        window_size=WINDOW_SIZE,
        max_torque=max_torque,
        include_time=INCLUDE_TIME,
    ),
    best_model_save_path=os.path.join(log_dir, "best_model"),
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

###############################################################################

agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    learning_rate=learning_rate,
)

setproctitle(
    f"noisy_training (domain and noise randomization) -> robot={robot} max_torque={max_torque}Nm robustness={robustness}"
)

agent.learn(total_timesteps=training_steps, callback=eval_callback)
