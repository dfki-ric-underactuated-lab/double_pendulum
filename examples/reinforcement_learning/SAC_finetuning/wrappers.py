import gymnasium as gym
from collections import deque
import numpy as np


class StateActionHistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length=4):
        super().__init__(env)
        self.history_length = history_length
        self.state_history = np.zeros((history_length, *env.observation_space.shape))
        self.action_history = np.zeros((history_length, *env.action_space.shape))

        state_space = env.observation_space
        action_space = env.action_space
        low = np.concatenate(
            [state_space.low] * history_length + [action_space.low] * history_length
        )
        high = np.concatenate(
            [state_space.high] * history_length + [action_space.high] * history_length
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        observation = self.env.reset()
        self.state_history.fill(0)
        self.state_history[-1] = observation[0]
        self.action_history.fill(0)
        return self._get_observation(), {}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.state_history[0:-1] = self.state_history[1:]
        self.state_history[-1] = observation
        self.action_history[0:-1] = self.action_history[1:]
        self.action_history[-1] = action
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        state_flat = self.state_history.reshape((-1))
        action_flat = self.action_history.reshape((-1))
        return np.concatenate([state_flat, action_flat])


# adds noise to velocity
class PerturbStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, perturb_std=0.02):
        super().__init__(env)
        self.std = perturb_std

    def observation(self, observation):
        observation = np.atleast_2d(observation)
        observation[:, 2:] = (
            observation[:, 2:] + np.random.randn(*observation[:, 2:].shape) * self.std
        )
        return observation


# probably useless since it's already perturbed by SAC_main_training
class PerturbActionWrapper(gym.ActionWrapper):
    def __init__(self, env, perturb_std=0.01):
        super().__init__(env)
        self.noise = lambda: np.random.randn(*env.observation_space.shape) * perturb_std

    def observation(self, observation):
        return observation + self.noise()


class TimeAwareWrapper(gym.ObservationWrapper):
    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt
        observation_space = env.observation_space
        low = np.append(observation_space.low, 0.0)
        high = np.append(observation_space.high, 1.0)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.t = 0

    def observation(self, observation):
        return np.append(observation, self.t / 10)

    def step(self, action):
        self.t += self.dt
        return super().step(action)

    def reset(self, **kwargs):
        self.t = 0
        return super().reset(**kwargs)
