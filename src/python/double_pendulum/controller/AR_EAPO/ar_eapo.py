from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Generator, NamedTuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium.spaces as spaces
from gymnasium.spaces import Space
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    SquashedDiagGaussianDistribution,
)
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs, GymEnv
from stable_baselines3.common.utils import (
    get_schedule_fn,
    obs_as_tensor,
    explained_variance,
)
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.ppo import PPO

from double_pendulum.controller.AR_EAPO import utils


@dataclass
class EAPOConfig:
    use_entropy_advantage: bool = True
    tau: float = 1.0
    c2: float = 0.5
    e_gamma: float | None = None
    e_lambda: float | None = None
    tau_on_entropy: bool = True
    use_shared_entropy_net: bool = True
    use_squashed_gaussian: bool = True


@dataclass
class ARConfig:
    use_weighted_average: bool = True
    r_step_size: float = 0.001
    use_trace_for_weighted_average: bool = True
    use_advantage: bool = True


class RolloutBufferSamplesWithEntropy(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    entropy_advantages: th.Tensor
    entropy_returns: th.Tensor


class ARRolloutBufferWithEntropy(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: Space,
        action_space: Space,
        device: Any | str = "auto",
        gae_lambda: float = 0.8,
        gamma: float = 1.0,
        n_envs: int = 1,
        e_lambda: float | None = None,
        e_gamma: float | None = None,
        ar_config: ARConfig = ARConfig(),
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs,
        )

        self.e_gamma = e_gamma if e_gamma is not None else self.gamma
        self.e_lambda = e_lambda if e_lambda is not None else self.gae_lambda
        self.use_weighted_average = ar_config.use_weighted_average
        self.step_size = ar_config.r_step_size
        self.use_trace = ar_config.use_trace_for_weighted_average
        self.use_advantage = ar_config.use_advantage
        self.rho_r = 0.0
        self.rho_e = 0.0
        self.trace = 0.0

    def reset(self) -> None:
        super().reset()
        self.entropies = np.zeros_like(self.rewards)
        self.entropy_predictions = np.zeros_like(self.values)
        self.entropy_advantage = np.zeros_like(self.advantages)
        self.entropy_returns = np.zeros_like(self.returns)
        self.bootstrap_rewards = np.zeros_like(self.rewards)
        self.bootstrap_entropies = np.zeros_like(self.rewards)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        entropy: np.ndarray,
        bootstrap_reward: np.ndarray,
        bootstrap_entropy: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        entropy_prediction: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        self.entropies[self.pos] = entropy
        self.entropy_predictions[self.pos] = entropy_prediction.cpu().numpy().flatten()
        self.bootstrap_rewards[self.pos] = bootstrap_reward
        self.bootstrap_entropies[self.pos] = bootstrap_entropy

        super().add(
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob,
        )

    def compute_returns_and_advantage(
        self,
        last_values: th.Tensor,
        last_entropy_predictions: th.Tensor,
        dones: np.ndarray,
    ) -> None:
        if not self.use_advantage:
            if self.use_weighted_average:
                step_size = self.step_size
                if self.use_trace:
                    self.trace = self.trace + step_size * (1 - self.trace)
                    step_size /= self.trace

                self.rho_r = self.rho_r + step_size * (self.rewards.mean() - self.rho_r)
                self.rho_e = self.rho_e + step_size * (
                    self.entropies.mean() - self.rho_e
                )
            else:
                self.rho_r = self.rewards.mean()
                self.rho_e = self.entropies.mean()

        self.rewards += -self.rho_r + self.bootstrap_rewards
        self.entropies += -self.rho_e + self.bootstrap_entropies

        super().compute_returns_and_advantage(last_values, dones)

        # Compute Entropy returns and advantages.
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            next_non_terminal: np.ndarray
            next_entropy: np.ndarray
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_entropy = last_entropy_predictions.clone().cpu().numpy().flatten()
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_entropy = self.entropy_predictions[step + 1]
            delta = (
                self.entropies[step]
                + self.e_gamma * next_entropy * next_non_terminal
                - self.entropy_predictions[step]
            )

            last_gae_lam = (
                delta + self.e_gamma * self.e_lambda * next_non_terminal * last_gae_lam
            )
            self.entropy_advantage[step] = last_gae_lam

        self.entropy_returns = self.entropy_advantage + self.entropy_predictions

        if self.use_advantage:
            step_size = self.step_size
            if self.use_trace:
                self.trace = self.trace + step_size * (1 - self.trace)
                step_size /= self.trace

            self.rho_r = self.rho_r + step_size * (self.advantages.mean())
            self.rho_e = self.rho_e + step_size * (self.entropy_advantage.mean())

    def get(
        self, batch_size: int | None = None
    ) -> Generator[RolloutBufferSamplesWithEntropy, None, None]:
        if not self.generator_ready:
            self.entropy_predictions = self.swap_and_flatten(self.entropy_predictions)
            self.entropy_returns = self.swap_and_flatten(self.entropy_returns)
            self.entropy_advantage = self.swap_and_flatten(self.entropy_advantage)
        return super().get(batch_size)  # type: ignore

    def _get_samples(
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> RolloutBufferSamplesWithEntropy:
        sample = super()._get_samples(batch_inds, env)
        return RolloutBufferSamplesWithEntropy(
            *sample,
            self.to_torch(self.entropy_advantage[batch_inds].flatten()),
            self.to_torch(self.entropy_returns[batch_inds].flatten()),
        )


class Policy(ActorCriticPolicy):
    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -20.0
    action_dist: Distribution

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        lr_schedule: Schedule,
        use_shared_entropy_net: bool = True,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn_pi: type[nn.Module] = nn.Tanh,
        activation_fn_vf: type[nn.Module] = nn.ReLU,
        use_squashed_gaussian: bool = True,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[optim.Optimizer] = optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        self.use_shared_entropy_net = use_shared_entropy_net
        self.activation_fn_pi = activation_fn_pi
        self.activation_fn_vf = activation_fn_vf
        self.squashed_gaussian = use_squashed_gaussian

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn_pi,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self._squash_output = use_squashed_gaussian

    class CustomMLPExtractor(nn.Module):
        def __init__(
            self,
            feature_dim: int,
            net_arch: dict[str, list[int]],
            activation_fn_pi: type[nn.Module],
            activation_fn_vf: type[nn.Module],
        ):
            super().__init__()
            self.policy_net = utils.build_mlp(
                feature_dim, None, net_arch["pi"], activation_fn_pi
            )
            self.value_net = utils.build_mlp(
                feature_dim, None, net_arch["vf"], activation_fn_vf
            )
            self.latent_dim_pi = net_arch["pi"][-1]
            self.latent_dim_vf = net_arch["vf"][-1]

        def forward_actor(self, features: th.Tensor) -> th.Tensor:
            return self.policy_net(features)

        def forward_critic(self, features: th.Tensor) -> th.Tensor:
            return self.value_net(features)

        def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
            return self.forward_actor(features), self.forward_critic(features)

    def _build_mlp_extractor(self) -> None:
        assert isinstance(self.net_arch, dict)
        self.mlp_extractor = Policy.CustomMLPExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn_pi=self.activation_fn_pi,
            activation_fn_vf=self.activation_fn_vf,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        action_dim = get_action_dim(self.action_space)

        # Build the corresponding policy head given the action distribtuion.
        if self.squashed_gaussian:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.log_std = None
            self.action_net = nn.Linear(latent_dim_pi, action_dim)
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if not self.use_shared_entropy_net:
            assert isinstance(self.net_arch, dict) and "vf" in self.net_arch
            self.entropy_features_extractor = utils.build_mlp(
                self.features_dim, None, self.net_arch["vf"], self.activation_fn
            )
        else:
            self.entropy_features_extractor = None

        self.entropy_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        self.log_std_net = None
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.log_std_net = nn.Linear(
                self.mlp_extractor.latent_dim_pi, self.action_dist.action_dim
            )
            nn.init.normal_(self.log_std_net.weight, std=0.01)
            with th.no_grad():
                self.log_std_net.bias.zero_()
                self.log_std_net.bias += np.log(np.exp(np.exp(self.log_std_init)) - 1)
            delattr(self, "log_std")

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.entropy_net: 1,
            }

            if self.features_extractor_class == NatureCNN:
                if not self.share_features_extractor:
                    # Note(antonin): this is to keep SB3 results
                    # consistent, see GH#1148
                    module_gains[self.pi_features_extractor] = np.sqrt(2)
                    module_gains[self.vf_features_extractor] = np.sqrt(2)
                else:
                    module_gains[self.features_extractor] = np.sqrt(2)

                if self.entropy_features_extractor is not None:
                    module_gains[self.entropy_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        if self.log_std_net is not None:
            self.log_std = th.log(F.softplus(self.log_std_net(latent_pi)))
            self.log_std = th.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return super()._get_action_dist_from_latent(latent_pi)

    def forward(self, obs: PyTorchObs, deterministic: bool = False):
        if self.share_features_extractor:
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = self.extract_features(obs)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        latent_ent: th.Tensor = (
            self.entropy_features_extractor(features)
            if self.entropy_features_extractor is not None
            else latent_vf
        )

        # Policy
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        # Value & Entropy
        values = self.value_net.forward(latent_vf)
        entropy_predictions = self.entropy_net.forward(latent_ent)

        # Entropy
        entropies = -log_prob

        return actions, values, entropy_predictions, log_prob, entropies

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor):
        if self.share_features_extractor:
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = self.extract_features(obs)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        latent_ent: th.Tensor = (
            self.entropy_features_extractor(features)
            if self.entropy_features_extractor is not None
            else latent_vf
        )

        # Policy
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Value & Entropy
        values = self.value_net.forward(latent_vf)
        entropy_predictions = self.entropy_net.forward(latent_ent)

        return values, entropy_predictions, log_prob, entropy

    def predict_values(self, obs: PyTorchObs):
        vf_features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(vf_features)
        latent_ent: th.Tensor = (
            self.entropy_features_extractor(vf_features)
            if self.entropy_features_extractor is not None
            else latent_vf
        )

        return self.value_net.forward(latent_vf), self.entropy_net.forward(latent_ent)


class AR_EAPO(PPO):
    rollout_buffer: ARRolloutBufferWithEntropy
    policy: Policy

    def __init__(
        self,
        env: GymEnv,
        ar_config: ARConfig = ARConfig(),
        eapo_config: EAPOConfig = EAPOConfig(),
        learning_rate: float | Schedule = 0.001,
        n_steps: int = 128,
        batch_size: int = 1024,
        n_epochs: int = 6,
        gamma: float = 1.0,
        gae_lambda: float = 0.8,
        clip_range: float | Schedule = 0.05,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0,
        vf_coef: float = 0.25,
        max_grad_norm: float = 10.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        policy: type[Policy] = Policy,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["use_shared_entropy_net"] = eapo_config.use_shared_entropy_net
        policy_kwargs["use_squashed_gaussian"] = eapo_config.use_squashed_gaussian
        if activation_pi := policy_kwargs.get("activation_fn_pi"):
            if isinstance(activation_pi, str):
                policy_kwargs["activation_fn_pi"] = utils.parse_activation_fn(
                    activation_pi
                )
        if activation_vf := policy_kwargs.get("activation_fn_vf"):
            if isinstance(activation_vf, str):
                policy_kwargs["activation_fn_vf"] = utils.parse_activation_fn(
                    activation_vf
                )

        rollout_buffer_kwargs = {
            "e_gamma": eapo_config.e_gamma,
            "e_lambda": eapo_config.e_lambda,
            "ar_config": ar_config,
        }

        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            ARRolloutBufferWithEntropy,  # rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            False,  # _init_setup_model,
        )

        self.ar_config = ar_config
        self.tau = eapo_config.tau
        self.c2 = eapo_config.c2
        self.e_gamma = (
            eapo_config.e_gamma if eapo_config.e_gamma is not None else self.gamma
        )
        self.e_lambda = (
            eapo_config.e_lambda
            if eapo_config.e_lambda is not None
            else self.gae_lambda
        )
        self.tau_on_entropy = eapo_config.tau_on_entropy
        self.disable_entropy_advantage = not eapo_config.use_entropy_advantage

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        OnPolicyAlgorithm._setup_model(self)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: ARRolloutBufferWithEntropy,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore
                actions, values, entropy_values, log_probs, entropies = (
                    self.policy.forward(obs_tensor)
                )
            actions = actions.cpu().numpy()
            entropies = entropies.cpu().numpy()

            if self.tau_on_entropy:
                entropies *= self.tau

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            bootstrap_rewards = np.zeros_like(rewards)
            bootstrap_entropies = np.zeros_like(entropies)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value, terminal_ent_value = self.policy.predict_values(terminal_obs)  # type: ignore[arg-type]

                    bootstrap_rewards[idx] = self.gamma * terminal_value.item()
                    bootstrap_entropies[idx] = self.e_gamma * terminal_ent_value.item()

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                entropies,
                bootstrap_rewards,
                bootstrap_entropies,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                entropy_values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values, entropy_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, last_entropy_predictions=entropy_values, dones=dones
        )

        self.logger.record("rollout/ρ_r", rollout_buffer.rho_r)
        self.logger.record("rollout/ρ_e", rollout_buffer.rho_e)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, entropy_value_losses = [], [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, entropy_values, log_prob, entropy = (
                    self.policy.evaluate_actions(rollout_data.observations, actions)
                )
                values = values.flatten()
                entropy_values = entropy_values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages

                if not self.disable_entropy_advantage:
                    if self.tau_on_entropy:
                        advantages += rollout_data.entropy_advantages
                    else:
                        advantages += self.tau * rollout_data.entropy_advantages

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf  # type: ignore
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(
                    rollout_data.returns.to(values_pred.dtype), values_pred
                )
                value_losses.append(value_loss.item())

                # Entropy critic loss
                if not self.disable_entropy_advantage:
                    entropy_value_loss = F.mse_loss(
                        rollout_data.entropy_returns.to(entropy_values.dtype),
                        entropy_values,
                    )
                    entropy_value_losses.append(entropy_value_loss.item())
                    value_loss = value_loss + self.c2 * entropy_value_loss

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Calculate explained variances
        values = self.rollout_buffer.values.flatten()
        explained_var = explained_variance(
            values, self.rollout_buffer.returns.flatten()
        )
        if not self.disable_entropy_advantage:
            entropy_values = self.rollout_buffer.entropy_predictions.flatten()
            entropy_explained_var = explained_variance(
                entropy_values,
                self.rollout_buffer.entropy_returns.flatten(),
            )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/entropy_critic_loss", np.mean(entropy_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if not self.disable_entropy_advantage:
            self.logger.record(
                "train/entropy_explained_variance", entropy_explained_var
            )
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())  # type: ignore

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def save(
        self,
        path: str | Path,
        exclude=None,
        include=None,
    ) -> None:
        super().save(path, exclude, include)
        if (vec_normalize_env := self.get_vec_normalize_env()) is not None:
            parent_dir = Path(path).parent
            vec_normalize_env.save(str(parent_dir / "vec_normalize_env.pkl"))

    @classmethod
    def load(
        cls: type["AR_EAPO"],
        path: str | Path,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "AR_EAPO":
        model = super().load(
            path, env, device, custom_objects, print_system_info, force_reset, **kwargs
        )
        if (vec_normalize_path := Path(path).parent / "vec_normalize_env.pkl").exists():
            import pickle

            with vec_normalize_path.open("rb") as f:
                vec_normalize = pickle.load(f)
                model._vec_normalize_env = vec_normalize

        return model
