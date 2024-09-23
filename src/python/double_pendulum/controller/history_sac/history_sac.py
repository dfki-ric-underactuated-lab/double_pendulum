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
from torch import nn

import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.policies import ContinuousCritic
import copy

from double_pendulum.controller.history_sac.utils import find_index_and_dict, get_state_values, softmax_and_select, load_param, default_decider
from double_pendulum.simulation.gym_env import CustomEnv
import torch.nn.functional as F


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


class SequenceExtractor(BaseFeaturesExtractor):
    """
        A feature extractor that processes sequences of observations. It separates additional features
        from the main feature set, processes them separately, and then combines them.

        Attributes:
            input_features (int): The dimensionality of the input features.
            timesteps (int): The timesteps in the sequence.
            output_dim (int): The output dimensionality after processing the features.
            additional_features (int): Number of additional features in the input.
    """
    def __init__(self, observation_space: gym.spaces.Box, translator):
        """
            Initializes the SequenceExtractor with the observation space and translator.

            Args:
                observation_space (gym.spaces.Box): The observation space of the environment.
                translator: An object that provides feature dimensionality and other configurations.
        """
        super().__init__(observation_space, translator.output_dim + translator.additional_features)
        self.input_features = translator.feature_dim
        self.timesteps = translator.timesteps
        self.output_dim = translator.output_dim
        self.additional_features = translator.additional_features

    def _process_additional_features(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
            Separates additional features from the main observation data.

            Args:
                obs (th.Tensor): The input observation tensor.

            Returns:
                tuple: A tuple containing:
                    - Additional features tensor if present.
                    - The remaining observation tensor.
        """
        if self.additional_features > 0:
            obs_1 = obs[:, :self.additional_features]
            obs_2 = obs[:, self.additional_features:]
            return obs_1, obs_2
        return None, obs

    def _combine_output(self, obs_1: th.Tensor, processed_output: th.Tensor) -> th.Tensor:
        """
            Combines the additional features with the processed main features.

            Args:
                obs_1 (th.Tensor): Additional features tensor.
                processed_output (th.Tensor): Processed main features tensor.

            Returns:
                th.Tensor: The combined tensor.
        """
        if obs_1 is not None:
            return th.cat((obs_1, processed_output), dim=1)
        return processed_output

    def forward(self, obs: th.Tensor) -> th.Tensor:
        obs_1, obs_2 = self._process_additional_features(obs)
        processed_output = self._process_main_features(obs_2)
        return self._combine_output(obs_1, processed_output)

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


class ConvExtractor(SequenceExtractor):
    def __init__(self, observation_space: gym.spaces.Box, translator, num_filters=12, num_heads=0, dropout=0.0):
        super().__init__(observation_space, translator)

        self.num_heads = num_heads

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(self.input_features, num_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)

        if num_heads > 0:
            # Multi-head self-attention
            self.self_attn = nn.MultiheadAttention(num_filters, num_heads, dropout=dropout)
        else:
            self.self_attn = None

        # Feature combination layers
        self.fc1 = nn.Linear(num_filters * self.timesteps, 256)
        self.fc2 = nn.Linear(256, self.output_dim)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_filters) if num_heads > 0 else None

    def _process_main_features(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.size(0)
        # Reshape the input tensor to have the shape (batch_size, timesteps, input_features)
        x = obs.view(batch_size, self.timesteps, self.input_features)
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, sequence_length)

        # Apply 1D convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.self_attn is not None:
            # Apply self-attention
            x = x.transpose(1, 2)  # (batch_size, sequence_length, num_filters)
            x = self.layer_norm(x) if self.layer_norm is not None else x
            attn_output, _ = self.self_attn(x, x, x)
            x = x + attn_output
            x = x.transpose(1, 2)  # (batch_size, num_filters, sequence_length)

        # Combine features
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)

        return x

class MultiplePoliciesReplayBuffer(ReplayBuffer):
    pass

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

class ConvTranslator(DefaultTranslator):
    """
        SequenceTranslator is responsible for preparing and translating observations from an environment
        into a format suitable for input into a neural network model. It handles the conversion of sequences
        of observations and actions, along with additional features, into a structured state representation.

        Attributes:
            timesteps (int): the timesteps to start the sequence.
            feature_dim (int): Dimensionality of the feature vector at each timestep.
            output_dim (int): Dimensionality of the output feature vector.
            additional_features (int): Number of additional features to include in the state representation.
            net_arch (list): Architecture of the neural network, specifying the number of units in each layer.
    """
    def __init__(self):
        self.reset()
        self.timesteps = 12
        self.feature_dim = 2
        self.output_dim = 8
        self.additional_features = 4
        self.net_arch = [1024, 1024, 1024]

        super().__init__(self.timesteps * self.feature_dim + self.additional_features)

    def build_state(self, observation, env) -> np.ndarray:
        """
            Builds the state representation from the current observation and environment state.

            Args:
                observation (object): The current observation from the environment.
                env (object): The environment instance providing the observation.

            Returns:
                np.ndarray: A flattened array containing the processed state representation.
        """

        index, history = find_index_and_dict(observation, env)
        dirty_observation = observation
        sequence_start = max(0, index + 1 - self.timesteps)

        X_meas = np.array(history['X_meas'])
        conv_memory = np.hstack((X_meas[sequence_start:index + 1, :-2],))

        if index < 0:
            print("This should not happen :(")

        output = conv_memory
        if output.shape[0] < self.timesteps:
            padding = np.zeros((self.timesteps - output.shape[0], output.shape[1]))
            output = np.vstack((padding, output))

        output = np.append(dirty_observation, output.flatten())

        return output

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


class ConvPolicy(CustomPolicy):
    """
        A custom Soft Actor-Critic (SAC) policy class that utilizes a sequence-based feature extractor (LSTM) for handling temporal dependencies in observations.

        This policy class extends the CustomPolicy class and integrates a SequenceTranslator for handling sequences of observations.
        It is specifically designed for environments where the temporal aspect of the data is crucial, such as in reinforcement learning
        tasks involving dynamic systems like robotics or control systems.

        Attributes:
        -----------
        translator : SequenceTranslator
            An instance of the SequenceTranslator class used for translating observations.

        additional_actor_kwargs : dict
            Additional keyword arguments for configuring the actor network, including its architecture.

        additional_critic_kwargs : dict
            Additional keyword arguments for configuring the critic network, including its architecture.

    """
    @classmethod
    def get_translator(cls) -> ConvTranslator:
        """
            A class method that returns an instance of the SequenceTranslator class. This translator is responsible for handling
            the preprocessing and translation of observations into a format suitable for the LSTM feature extractor.
        """
        return ConvTranslator()

    def __init__(self, *args, **kwargs):
        """
            Initializes the SequenceSACPolicy.

            Sets up the SequenceTranslator for the policy, configures the network architecture for the actor and critic networks,
            and initializes the feature extractor with an LSTM-based extractor.

            Parameters:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
        """
        self.translator = self.get_translator()
        self.additional_actor_kwargs['net_arch'] = self.translator.net_arch
        self.additional_critic_kwargs['net_arch'] = self.translator.net_arch

        kwargs.update(
            dict(
                features_extractor_class=ConvExtractor,
                features_extractor_kwargs=dict(translator=self.translator),
                share_features_extractor=False,
                optimizer_kwargs={'weight_decay': 0.000001}
            )
        )

        super().__init__(*args, **kwargs)

    def after_critic_backward(self):
        pass
        # th.nn.utils.clip_grad_norm_(self.critic.parameters(), 25)

class CustomUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if "examples" in module:
            module = "double_pendulum.controller.history_sac.history_sac"
        return super().find_class(module, name)


def custom_load(path):
    print(path)
    with open(path, 'rb') as f:
        print(f)
        return CustomUnpickler(f).load()


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
        loaded_data = custom_load(os.path.abspath(path + '.pkl'))

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
