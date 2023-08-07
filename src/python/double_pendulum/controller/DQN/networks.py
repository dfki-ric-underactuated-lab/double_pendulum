from typing import Tuple, Sequence
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp

from double_pendulum.controller.DQN.replay_buffer import ReplayBuffer
from double_pendulum.controller.DQN.utils import save_pickled_data


class BaseQ:
    def __init__(
        self,
        q_inputs: dict,
        n_actions: int,
        gamma: float,
        network: nn.Module,
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.network = network
        self.network_key = network_key
        self.params = self.network.init(self.network_key, **q_inputs)
        self.target_params = self.params
        self.n_training_steps_per_online_update = n_training_steps_per_online_update

        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))

        if learning_rate is not None:
            self.optimizer = optax.adam(learning_rate)
            self.optimizer_state = self.optimizer.init(self.params)

    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        raise NotImplementedError

    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict, ord: int = 2) -> jnp.float32:
        raise NotImplementedError

    @staticmethod
    def metric(error: jnp.ndarray, ord: str) -> jnp.float32:
        if ord == "huber":
            return optax.huber_loss(error, 0).mean()
        elif ord == "1":
            return jnp.abs(error).mean()
        elif ord == "2":
            return jnp.square(error).mean()
        elif ord == "sum":
            return jnp.square(error).sum()

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state: Tuple, batch_samples: jnp.ndarray
    ) -> Tuple[FrozenDict, FrozenDict, jnp.float32]:
        loss, grad_loss = self.loss_and_grad(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer, key: jax.random.PRNGKeyArray) -> jnp.float32:
        if step % self.n_training_steps_per_online_update == 0:
            batch_samples = replay_buffer.sample_random_batch(key)

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )

            return loss
        else:
            return jnp.nan

    def update_target_params(self, step: int) -> None:
        raise NotImplementedError

    @partial(jax.jit, static_argnames="self")
    def random_action(self, key: jax.random.PRNGKeyArray) -> jnp.int8:
        return jax.random.choice(key, jnp.arange(self.n_actions)).astype(jnp.int8)

    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        raise NotImplementedError

    def save(self, path: str) -> None:
        save_pickled_data(path + "_online_params", self.params)


class DQN(BaseQ):
    def __init__(
        self,
        state_shape: list,
        n_actions: int,
        gamma: float,
        layers: Sequence[int],
        network_key: jax.random.PRNGKeyArray,
        learning_rate: float,
        n_training_steps_per_online_update: int,
        n_training_steps_per_target_update: int,
    ) -> None:
        super().__init__(
            {"state": jnp.zeros(state_shape, dtype=jnp.float32)},
            n_actions,
            gamma,
            MLP(layers + [n_actions]),
            network_key,
            learning_rate,
            n_training_steps_per_online_update,
        )
        self.n_training_steps_per_target_update = n_training_steps_per_target_update

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def compute_target(self, params: FrozenDict, samples: FrozenDict) -> jnp.ndarray:
        return samples["reward"] + (1 - samples["absorbing"]) * self.gamma * self.apply(
            params, samples["next_state"]
        ).max(axis=1)

    @partial(jax.jit, static_argnames="self")
    def loss(self, params: FrozenDict, params_target: FrozenDict, samples: FrozenDict) -> jnp.float32:
        targets = self.compute_target(params_target, samples)
        q_states_actions = self.apply(params, samples["state"])

        # mapping over the states
        predictions = jax.vmap(lambda q_state_actions, action: q_state_actions[action])(
            q_states_actions, samples["action"]
        )

        return self.metric(predictions - targets, ord="2")

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int8:
        # key is not used here
        return jnp.argmax(self.apply(params, jnp.array(state, dtype=jnp.float32))).astype(jnp.int8)

    def update_target_params(self, step: int) -> None:
        if step % self.n_training_steps_per_target_update == 0:
            self.target_params = self.params


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, state):
        x = state
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
