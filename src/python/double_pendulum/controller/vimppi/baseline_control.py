from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.typing as jtp
import numpy as np
from double_pendulum.controller.AR_EAPO import AR_EAPOController


def create_zero_control_generator(horizon: int, nu: int):
    @jax.jit
    def control_generator(x0: jnp.ndarray, xf: jnp.ndarray) -> jnp.ndarray:
        """
        Generate zero control signal.

        Args:
            x (jnp.ndarray): State
            t (float): Time
        Returns:
            jnp.ndarray: Zero control signal
        """
        return jnp.zeros((horizon, nu))

    return control_generator


def create_tv_lqr_control_generator(
    dyn_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    lqr_dt: float,
    mppi_horizon: int,
    mppi_dt: float,
    nu: int,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    Q_final: jnp.ndarray,
    u_min: float = -6.0,
    u_max: float = 6.0,
):
    # Get desired trajectory via cubic interpolation
    mppi2lqr_ratio = int(mppi_dt // lqr_dt)
    lqr_horizon = mppi_horizon * mppi2lqr_ratio

    # TODO: is this correct?
    @jax.jit
    def x_dot(x, u):
        return jnp.concatenate([x[2:], dyn_fn(x, u)])

    Jx_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.jacobian(lambda x: x_dot(x, jnp.zeros(nu)), argnums=0)
    # Ju_fn = B

    @jax.jit
    def contorl_generator(
        x0: jnp.ndarray,
        xf: jnp.ndarray,
    ) -> jnp.ndarray:
        # Generate cubic interpolation trajectory
        T = lqr_horizon * lqr_dt
        A_least_squares = jnp.array(
            [
                [1, 0, 0, 0],
                [1, T, T**2, T**3],
                [0, 1, 0, 0],
                [0, 1, 2 * T, 3 * T**2],
            ]
        )

        ts_values = jnp.arange(0, T, lqr_dt)
        q_parts = []
        v_parts = []
        num = x0[:2].shape[0]
        for i in range(num):
            q0_i = x0[:2][i]
            v0_i = x0[2:][i]
            q_des_i = xf[:2][i]
            v_des_i = xf[2:][i]
            b_least_squares = jnp.array([q0_i, q_des_i, v0_i, v_des_i])
            a = jnp.linalg.solve(A_least_squares, b_least_squares)
            q_i = a[0] + a[1] * ts_values + a[2] * ts_values**2 + a[3] * ts_values**3
            v_i = a[1] + 2 * a[2] * ts_values + 3 * a[3] * ts_values**2
            q_parts.append(q_i)
            v_parts.append(v_i)
        traj = jnp.vstack(q_parts + v_parts).T
        # FIXME: the velocity is too high (12 r/s)

        # Linear System Discretization
        A_d, B_d = jax.vmap(lambda traj: (jnp.eye(len(x0)) + Jx_fn(traj) * lqr_dt, B * lqr_dt), in_axes=0)(traj)

        def dlqr_ltv(P, state):
            A, B = state
            K = jnp.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
            return P, K

        # Compute LQR gain
        _, K = jax.lax.scan(dlqr_ltv, Q_final, (A_d[::-1], B_d[::-1]))
        K = K[::-1]

        # Rollout the dynamics for the horizon
        def dyn_scan(x, input):
            K, x_des = input
            u = -K @ (x - x_des)
            u = jnp.clip(u, u_min, u_max)
            dxdt = x_dot(x, u)

            return x + dxdt * lqr_dt, u

        x_last, us = jax.lax.scan(dyn_scan, x0, (K, traj))
        # FIXME: should we skip the first control?
        return us[1::mppi2lqr_ratio]

    return contorl_generator


def create_ar_eapo_generator(
    step_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    horizon: int,
    dt: float,
    model_path: str,
    robot: str,
):
    model_path = Path(model_path)
    controller = AR_EAPOController(
        model_path=model_path,
        robot=robot,
        max_torque=6.0,
        max_velocity=20.0,
        deterministic=True,
    )

    def control_generator(x0: jnp.ndarray, _: jnp.ndarray) -> jnp.ndarray:
        us = []
        for _ in range(horizon):
            u = controller.get_control_output(x0)
            x0 = step_fn(dt, x0, u)
            us.append(u)
        # print("CONVERGED TO: ", x0)
        return jnp.array(us)

    return control_generator
