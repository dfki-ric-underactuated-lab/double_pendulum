import os
from typing import Tuple, Dict
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import jax
import jax.numpy as jnp

from double_pendulum.controller.DQN.replay_buffer import ReplayBuffer
from double_pendulum.controller.DQN.networks import BaseQ
from double_pendulum.controller.DQN.exploration import EpsilonGreedySchedule
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import (
    CustomEnv,
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_diff


# define robot variation
# robot = "acrobot"
robot = "pendubot"

# model parameter
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    design = "design_A.0"
    model = "model_2.0"
    load_path = "../../../examples/reinforcement_learning/DQN/lqr_data/pendubot/lqr/roa"

elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    design = "design_C.0"
    model = "model_3.0"
    load_path = "../../../examples/reinforcement_learning/DQN/lqr_data/acrobot/lqr/roa"

model_par_path = (
    "../../../data/system_identification/identified_parameters/" + design + "/" + model + "/model_parameters.yml"
)

mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)
dt = 0.01
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 2
obs_space = gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
max_steps = 1000
termination = False

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
)

# import lqr parameters
rho = np.loadtxt(os.path.join(load_path, "rho"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))


def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad


def reward_func(observation, action):
    # quadratic with roa attraction
    Q = np.zeros((4, 4))
    Q[0, 0] = 10
    Q[1, 1] = 10
    Q[2, 2] = 0.4
    Q[3, 3] = 0.3
    R = np.array([[0.0001]])

    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * 8.0,
            observation[3] * 8.0,
        ]
    )

    u = 5.0 * action

    goal = [np.pi, 0.0, 0.0, 0.0]

    y = wrap_angles_diff(s)

    flag = False
    bonus = False

    # openAI
    p1 = y[0]
    p2 = y[1]
    ee1_pos_y = -0.2 * np.cos(p1)

    ee2_pos_y = ee1_pos_y - 0.3 * np.cos(p1 + p2)

    control_line = 0.4

    # criteria 2
    bonus, _ = check_if_state_in_roa(S, rho, y)

    # criteria 4
    if ee2_pos_y >= control_line:
        flag = True
    else:
        flag = False

    r = np.einsum("i, ij, j", s - goal, Q, s - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r
    if flag:
        reward += 100
        if bonus:
            # roa method
            reward += 1e3
            # print("!!!bonus = True")
    else:
        reward = reward

    return reward


def terminated_func(observation):
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * 8.0,
            observation[3] * 8.0,
        ]
    )
    # y = wrap_angles_top(s)
    y = wrap_angles_diff(s)
    bonus, rad = check_if_state_in_roa(S, rho, y)
    if termination:
        if bonus:
            print("terminated")
            return bonus
    else:
        return False


def noisy_reset_func():
    rand = np.random.rand(4) * 0.01
    rand[2:] = rand[2:] - 0.05
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation


class Env(CustomEnv):
    def __init__(
        self,
        actions,
        dynamics_func,
        reward_func,
        terminated_func,
        reset_func,
        obs_space,
        act_space,
        max_episode_steps,
    ):
        super().__init__(
            lambda state, action: dynamics_func(state, np.array([actions[action]])),
            lambda state, action: reward_func(state, np.array([actions[action]])),
            terminated_func,
            reset_func,
            obs_space,
            act_space,
            max_episode_steps,
        )
        self.actions = actions
        self.n_actions = len(actions)

    def step(self, action):
        self.observation = self.dynamics_func(self.observation, action)
        reward = self.reward_func(self.observation, action)
        terminated = self.terminated_func(self.observation)

        self.step_counter += 1

        return self.observation, reward, terminated, {}

    def reset(self):
        self.step_counter = 0

        return super().reset()

    def collect_random_samples(
        self, sample_key: jax.random.PRNGKeyArray, replay_buffer: ReplayBuffer, n_samples: int, horizon: int
    ) -> None:
        self.reset()

        for _ in tqdm(range(n_samples)):
            observation = self.observation

            sample_key, key = jax.random.split(sample_key)
            action = jax.random.choice(key, jnp.arange(self.n_actions))
            next_observation, reward, terminated, _ = self.step(action)

            replay_buffer.add(observation, action, reward, next_observation, terminated)

            if terminated or self.step_counter >= horizon:
                self.reset()

    def collect_one_sample(
        self,
        q: BaseQ,
        q_params: Dict,
        horizon: int,
        replay_buffer: ReplayBuffer,
        exploration_schedule: EpsilonGreedySchedule,
    ) -> Tuple[float, bool]:
        state = self.observation

        if exploration_schedule.explore():
            action = q.random_action(exploration_schedule.key)
        else:
            action = q.best_action(q_params, self.observation, exploration_schedule.key)

        next_state, reward, absorbing, _ = self.step(action)

        replay_buffer.add(state, action, reward, next_state, absorbing)

        if absorbing or self.step_counter >= horizon:
            self.reset()

        return reward, absorbing or self.step_counter == 0


def get_environment(n_actions: int):
    return Env(
        np.hstack((-np.logspace(-1, 0, n_actions // 2)[::-1], np.zeros(1), np.logspace(-1, 0, n_actions // 2))),
        dynamics_func,
        reward_func,
        terminated_func,
        noisy_reset_func,
        obs_space,
        act_space,
        max_steps,
    )
