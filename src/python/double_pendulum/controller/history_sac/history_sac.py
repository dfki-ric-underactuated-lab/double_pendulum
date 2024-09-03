from stable_baselines3 import SAC
from typing import Union, Dict, Any, Tuple, Type
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
import pathlib
import io
import os
import torch as th
import pickle
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic
import copy

from double_pendulum.controller.history_sac.utils import find_index_and_dict, get_state_values, softmax_and_select, load_param, default_decider
from double_pendulum.simulation.gym_env import CustomEnv


class HistoryEnv(CustomEnv):

    def __init__(
        self,
        dynamic_function,
        reward_function,
        termination_function,
        reset_function,
        torque_limit
    ):
        self.mpar = load_param(torque_limit=torque_limit)
        self.history = {"T": [], 'X_meas': [], 'X_real': [], 'U_con': [], 'U_real': [], "push": [], "mpar": self.mpar}
        self.history_old = None
        self.reset_function = reset_function
        super().__init__(
            dynamic_function,
            reward_function,
            termination_function,
            self.custom_reset,
        )
        self.dynamics_func.simulator.plant.history = self.history

    def custom_reset(self):
        self.history_old = copy.deepcopy(self.history)
        if 'dynamics_func' not in self.history:
            self.history['dynamics_func'] = self.dynamics_func
        for key in self.history:
            if key != 'dynamics_func' and key != 'max_episode_steps' and key != 'mpar':
                self.history[key].clear()

        clean_observation = np.array(self.reset_function())
        self.append_history(clean_observation, clean_observation, 0.0)
        self.history['U_con'].append(0.0)

        return clean_observation

    def append_history(self, clean_observation, dirty_observation, dirty_action: float):
        time = 0
        if len(self.history['T']) > 0:
            time = self.dynamics_func.dt + self.history['T'][-1]
        self.history['T'].append(np.round(time, decimals=5))
        self.history['U_real'].append(dirty_action)
        self.history['X_meas'].append(dirty_observation)
        self.history['X_real'].append(clean_observation)


class DefaultTranslator:

    def __init__(self, input_dim: int):
        self.obs_space = gym.spaces.Box(-np.ones(input_dim), np.ones(input_dim))
        self.act_space = gym.spaces.Box(np.array([-1.0]), np.array([1.0]))

    def build_state(self, observation, env) -> np.ndarray:
        """
            Processes and returns the latest raw observation from the environment's observation dictionary.

            This method retrieves the most recent measurement from the environment's observation dictionary and returns it.
        """
        dirty_observation = env.history['X_meas'][-1]
        return dirty_observation

    def reset(self):
        pass


class PastActionsTranslator(DefaultTranslator):
    def __init__(self):
        self.past_action_number = 0
        self.reset()
        super().__init__(8 + self.past_action_number)

    def build_state(self, observation, env) -> np.ndarray:
        index, history = find_index_and_dict(observation, env)
        dirty_observation = history['X_meas'][index]

        u_con = history['U_con']
        action_memory = np.zeros(self.past_action_number)
        actions_to_copy = min(index, len(u_con), self.past_action_number)

        if actions_to_copy > 0:
            action_memory[-actions_to_copy:] = u_con[-actions_to_copy:]

        state_values = get_state_values(history, offset=index + 1 - len(history['T']))
        l_ges = env.mpar.l[0] + env.mpar.l[1]
        additional = np.array([
            state_values['x2'][1] / l_ges,
            state_values['v2'][0] / env.dynamics_func.max_velocity,
            state_values['c1'],
            state_values['c2']
        ])

        state = np.append(additional, dirty_observation.copy())
        state = np.append(state, action_memory)

        return state


class MultiplePoliciesReplayBuffer(ReplayBuffer):
    pass


class CustomPolicy(SACPolicy):
    """
        A base class for custom Soft Actor-Critic (SAC) policies.

        This class extends the SACPolicy and serves as a template for more specialized policies. It includes additional mechanisms for
        handling actor and critic network configurations through keyword arguments, and it supports the use of a custom translator for preprocessing observations.
    """
    additional_actor_kwargs = {}
    additional_critic_kwargs = {}

    def __init__(self, *args, **kwargs):
        self.translator = self.get_translator()
        super().__init__(*args, **kwargs)

    @classmethod
    def get_translator(cls) -> DefaultTranslator:
        return DefaultTranslator(4)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs.update(self.additional_actor_kwargs)
        actor = Actor(**actor_kwargs).to(self.device)
        return actor

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update(self.additional_critic_kwargs)
        critic = ContinuousCritic(**critic_kwargs).to(self.device)
        return critic


class SequenceSACPolicy(CustomPolicy):
    pass


class PastActionsSACPolicy(CustomPolicy):

    @classmethod
    def get_translator(cls) -> PastActionsTranslator:
        return PastActionsTranslator()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HistorySAC(SAC):

    def __init__(self, policy_classes, replay_buffer_classes, *args, **kwargs):
        self.env_type = kwargs['env_type']

        self.policies = []
        self.replay_buffers = []

        self.replay_buffer_classes = replay_buffer_classes
        self.policy_classes = policy_classes
        self.policy_number = len(self.policy_classes)

        kwargs['policy'] = self.policy_classes[0]
        del kwargs['env_type']
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        for i in range(self.policy_number):
            if len(self.replay_buffer_classes) > i:
                self.replay_buffer_class = self.replay_buffer_classes[i]
            else:
                self.replay_buffer_class = ReplayBuffer
            self.policy_class = self.policy_classes[i]
            self.replay_buffer = None
            translator = self.policy_class.get_translator()
            self.observation_space = translator.obs_space
            self.action_space = translator.act_space

            layers = [512, 512, 512]
            if self.env_type == "acrobot":
                layers = [256, 512, 256]
            self.policy_class.additional_actor_kwargs['net_arch'] = layers
            self.policy_class.additional_critic_kwargs['net_arch'] = layers

            super()._setup_model()
            self.replay_buffers.append(self.replay_buffer)
            self.policies.append(self.policy)

    @classmethod
    def load(
            cls: Type[SelfBaseAlgorithm],
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,

    ) -> SelfBaseAlgorithm:
        loaded_data = th.load(os.path.abspath(path + '.pkl'))

        model = HistorySAC(
            policy_classes=loaded_data["policy_classes"],
            replay_buffer_classes=loaded_data["replay_buffer_classes"],
            env=env,
            **kwargs
        )

        for i, policy_state in enumerate(loaded_data["policies"]):
            model.policies[i].load_state_dict(policy_state)

        return model

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        return self.get_actions(observation, deterministic), None

    def get_actions(self, obs, deterministic):
        selected_policies = self.decide_policy(obs)
        envs = [m.env for m in self.env.envs]
        n_envs = len(envs)

        policy_indices = np.argmax(selected_policies, axis=0)
        states = [[] for _ in range(self.policy_number)]

        for policy_index, env_index in enumerate(policy_indices):
            states[env_index].append(self.policies[env_index].translator.build_state(obs[policy_index], envs[policy_index]))

        actions = np.empty((n_envs,) + self.policies[0].action_space.shape, dtype=self.policies[0].action_space.dtype)

        for policy_index in range(self.policy_number):
            if states[policy_index]:
                policy_actions, _ = self.policies[policy_index].predict(np.array(states[policy_index]), deterministic=deterministic)
                actions = policy_actions

        return actions

    def decide_policy(self, new_obs):
        decider = [lambda x, y: 1]
        assignments = np.array([
            [func(obs, 0) for obs in new_obs]
            for func in decider
        ])
        return softmax_and_select(assignments)
