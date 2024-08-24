import gymnasium as gym
import numpy as np
import math


class CustomEnv(gym.Env):
    def __init__(
        self,
        dynamics_func,
        reward_func,
        terminated_func,
        reset_func,
        obs_space=gym.spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
        ),
        act_space=gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        max_episode_steps=1000,
        scaling=True,
    ):
        self.dynamics_func = dynamics_func
        self.reward_func = reward_func
        self.terminated_func = terminated_func
        self.reset_func = reset_func

        self.observation_space = obs_space
        self.action_space = act_space
        self.max_episode_steps = max_episode_steps

        self.observation = self.reset_func()
        self.step_counter = 0
        self.scaling = scaling

    def step(self, action):
        self.observation = self.dynamics_func(
            self.observation, action, scaling=self.scaling
        )
        reward = self.reward_func(self.observation, action)
        terminated = self.terminated_func(self.observation)
        info = {}
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
            self.step_counter = 0
        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = self.reset_func()
        self.step_counter = 0
        info = {}
        return self.observation, info

    def render(self, mode="human"):
        pass


class double_pendulum_dynamics_func:
    def __init__(
        self,
        simulator,
        dt=0.01,
        integrator="runge_kutta",
        robot="double_pendulum",
        state_representation=2,
        max_velocity=20.0,
        torque_limit=[5.0, 5.0],
        scaling=True,
    ):
        self.simulator = simulator
        self.dt = dt
        self.integrator = integrator
        self.robot = robot
        self.state_representation = state_representation
        self.max_velocity = max_velocity

        self.torque_limit = torque_limit
        self.scaling = scaling

    def __call__(self, state, action, scaling=True):
        if scaling:
            x = self.unscale_state(state)
            u = self.unscale_action(action)
            xn = self.integration(x, u)
            obs = self.normalize_state(xn)
            return obs.copy()
        else:
            u = self.unscale_action(action)
            xn = self.integration(state, u)
            return xn.copy()

    def integration(self, x, u):
        if self.integrator == "runge_kutta":
            next_state = np.add(
                x,
                self.dt * self.simulator.runge_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        elif self.integrator == "euler":
            next_state = np.add(
                x,
                self.dt * self.simulator.euler_integrator(x, self.dt, 0.0, u),
                casting="unsafe",
            )
        return next_state

    def unscale_action(self, action):
        """
        scale the action
        [-1, 1] -> [-limit, +limit]
        """
        if self.robot == "double_pendulum":
            a = [
                float(self.torque_limit[0] * action[0]),
                float(self.torque_limit[1] * action[1]),
            ]
        elif self.robot == "pendubot":
            a = np.array([float(self.torque_limit[0] * action[0]), 0.0])
        elif self.robot == "acrobot":
            a = np.array([0.0, float(self.torque_limit[1] * action[0])])
        return a

    def unscale_state(self, observation):
        """
        scale the state
        [-1, 1] -> [-limit, +limit]
        """
        if self.state_representation == 2:
            x = np.array(
                [
                    observation[0] * np.pi + np.pi,
                    observation[1] * np.pi + np.pi,
                    observation[2] * self.max_velocity,
                    observation[3] * self.max_velocity,
                ]
            )
        elif self.state_representation == 3:
            x = np.array(
                [
                    np.arctan2(observation[0], observation[1]),
                    np.arctan2(observation[2], observation[3]),
                    observation[4] * self.max_velocity,
                    observation[5] * self.max_velocity,
                ]
            )
        return x

    def normalize_state(self, state):
        """
        rescale state:
        [-limit, limit] -> [-1, 1]
        """
        if self.state_representation == 2:
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
        elif self.state_representation == 3:
            observation = np.array(
                [
                    np.cos(state[0]),
                    np.sin(state[0]),
                    np.cos(state[1]),
                    np.sin(state[1]),
                    np.clip(state[2], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                    np.clip(state[3], -self.max_velocity, self.max_velocity)
                    / self.max_velocity,
                ]
            )

        return observation
