from collections.abc import Callable
import jax
import jax.numpy as jnp
import jax.typing as jpt


def create_time_disturbance_checker(
    threshold: float,
) -> Callable[[jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike], bool]:
    @jax.jit
    def check_disturbance(dt, x_prev, x, u):
        return dt > threshold

    return check_disturbance


def createt_state_disturbance_checker(
    threshold: float, step_fn
) -> Callable[[float, jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike], bool]:
    @jax.jit
    def check_disturbance(dt, x_prev, x, u):
        x_intergration_error = jnp.abs(step_fn(dt, x_prev, u) - x)

        return (x_intergration_error > threshold).any()

    return check_disturbance
