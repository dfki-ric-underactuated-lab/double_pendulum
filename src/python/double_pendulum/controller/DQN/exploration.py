from functools import partial
import jax
import optax


class EpsilonGreedySchedule:
    def __init__(
        self,
        starting_eps: float,
        ending_eps: float,
        duration_eps: int,
        key: jax.random.PRNGKeyArray,
        current_exploration_step: int,
    ) -> None:
        self.epsilon_schedule = optax.linear_schedule(starting_eps, ending_eps, duration_eps)
        self.current_exploration_step = current_exploration_step
        self.key = key

    def explore(self) -> bool:
        self.current_exploration_step += 1
        self.key, key = jax.random.split(self.key)

        return self.explore_(key, self.current_exploration_step)

    @partial(jax.jit, static_argnames="self")
    def explore_(self, key, exploration_step):
        return jax.random.uniform(key) < self.epsilon_schedule(exploration_step)
