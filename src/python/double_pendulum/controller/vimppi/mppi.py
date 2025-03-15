from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.typing as jpt

from .config import Config

# TODO: find appropriate place for this
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


def moving_average_filter(data: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """
    Apply a moving average filter to a multi-dimensional JAX array along the time axis.

    Args:
        data (jnp.ndarray): The input array to filter, shape (T, act_dim).
        window_size (int): The size of the moving average window.

    Returns:
        jnp.ndarray: The filtered array with the moving average applied along the time axis.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    # Create a kernel for the moving average
    kernel = jnp.ones(window_size) / window_size

    # Pad the data along the time axis (axis=0)
    padded_data = jnp.pad(data, ((window_size - 1, 0), (0, 0)), mode="edge")

    # Convolve along the time axis for each action dimension
    smoothed_data = jax.vmap(
        lambda col: jnp.convolve(col, kernel, mode="valid"), in_axes=0
    )(padded_data.T).T

    return smoothed_data


def create_mppi(cfg: Config) -> Callable:
    def lax_wrapper_step(carry, action):
        state, dt = carry[0], carry[1]
        next_state = cfg.step(dt, state, action)
        carry = (next_state, dt)
        return carry, next_state

    def build_rollout_fn(rollout_fn_step):
        def rollout_fn(state, control, dt):
            carry = (state, dt)
            _, states = jax.lax.scan(rollout_fn_step, carry, control)
            return states

        # vectorize rollout function
        func = jax.jit(jax.vmap(rollout_fn, in_axes=(None, 0, None)))
        return func

    rollout_fn = build_rollout_fn(lax_wrapper_step)
    sigma_inv = jnp.linalg.inv(cfg.sigma)
    gamma = cfg.lambda_ * (1.0 - cfg.alpha)

    @jax.jit
    def mppi(
        obs: jpt.ArrayLike, u_prev: jpt.ArrayLike, dt: float
    ) -> tuple[jpt.ArrayLike, jpt.ArrayLike]:
        # sample noise
        mu = jnp.zeros(cfg.act_dim)
        epsilon = jax.random.multivariate_normal(
            cfg.key, mu, cfg.sigma, (cfg.samples, cfg.horizon)
        )

        # compute control sequence with epsilon-noise perturbations
        # first (1.0 - exploration) * samples trajectories should exploit
        # other should explore
        # v = u_prev + epsilon
        v = jnp.zeros((cfg.samples, cfg.horizon, cfg.act_dim))
        explore_idx = int((1.0 - cfg.exploration) * cfg.samples)
        v = v.at[:explore_idx].set(u_prev + epsilon[:explore_idx])
        v = v.at[explore_idx:].set(epsilon[explore_idx:])

        # bound control sequence
        v = jnp.clip(v, cfg.act_min, cfg.act_max)

        # perform rollouts to find trajectories
        states = rollout_fn(obs, v, dt)

        # augment stage cost to include sigma term
        def stage_cost(x, u, u_prev):
            base_cost = cfg.stage_cost(x, u)
            augmented = gamma * u.T @ sigma_inv @ u_prev

            return base_cost + augmented

        # Compute stage costs for all samples and all horizons
        # stage_costs: shape (samples, horizon)
        stage_costs = jax.vmap(
            lambda trajectory, controls, prev_controls: jax.vmap(stage_cost)(
                trajectory,
                controls,
                prev_controls,
            ),
            in_axes=(0, 0, None),
        )(states, v, u_prev)

        # Accumulate stage costs for each trajectory
        # cumulative_stage_costs: shape (samples,)
        cumulative_stage_costs = jnp.sum(stage_costs, axis=1)

        # Compute terminal costs for all trajectories
        # terminal_costs: shape (samples,)
        terminal_costs = jax.vmap(cfg.terminal_cost)(states[:, -1])

        # Total cost per trajectory
        # S: shape (samples,)
        S = cumulative_stage_costs + terminal_costs

        # compute weights
        rho = S.min()
        eta = jnp.sum(jnp.exp((-1.0 / cfg.lambda_) * (S - rho)))
        w = (1.0 / eta) * jnp.exp((-1.0 / cfg.lambda_) * (S - rho))

        # calculate wk * epsilon_k
        w_epsilon = jnp.einsum("k, kij -> ij", w, epsilon)

        # apply moving average to control sequence
        w_epsilon = moving_average_filter(w_epsilon, 10)

        u = jnp.clip(u_prev + w_epsilon, cfg.act_min, cfg.act_max)
        next_action = jnp.clip(u[0], cfg.act_min, cfg.act_max)

        # create new control sequence that will be used in the next iteration
        a = 1 / 10
        u_next = u
        # Linear interpolation (from second to one before last)
        # u_next = u_next.at[1:-1].set(a * (u[2:] + (1 - a) * u[1:-1]))
        u_next = u_next.at[0:-1].add(a * (u[1:] - u[0:-1]))
        # u_next = jnp.zeros_like(u)
        # u_next = u_next.at[:-1].set(u[1:])
        # u_next = jnp.roll(u, -1, axis=0)
        # u_next = u_next.at[-1].set(jnp.zeros(cfg.act_dim))

        # return next control and next control sequence
        return next_action, u_next

    return mppi
