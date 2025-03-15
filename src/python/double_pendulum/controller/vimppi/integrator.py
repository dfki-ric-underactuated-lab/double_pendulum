"""
Integration module. Implements different integrators which are used as
step functions in MPPI rollouts. Currently implemented:

1) Forward Dynamics Explicit Euler
2) DELM Variational Integrator
"""

from functools import partial

import jax
import jax.numpy as jnp

from .plant import (
    D,
    D_fast,
    compute_discrete_force,
    discrete_lagrangian,
    mass_matrix,
)


def create_explicit_euler_integrator(fwddyn_fn):
    """
    Wrapper to formulate Explicit Euler integrator.

    Args:
        fwddyn_fn (Callable[jnp.ndarray, jnp.ndarray]): Forward dynamics function
        dt (float): Solution timestep
    Returns:
        Callable[jnp.ndarray, jnp.ndarray]: Step function, which computes the next state
                                            based on current state and control signal
    """

    def _explicit_euler(dt, x, u):
        """
        Explicit Euler implementation.

        Args:
            x (jnp.ndarray): Current state
            u (jnp.ndarray): Control signal
        Returns:
            jnp.ndarray: The new state
        """
        acc = fwddyn_fn(x, u)
        dx = jnp.array([x[2], x[3], acc[0], acc[1]])
        x = x + dt * dx
        return x

    return _explicit_euler


def create_implicit_integrator(mpar, fwddyn_fn):
    """
    Wrapper to formulate Implicit Euler integrator.
    """

    def _implicit(dt, x, u):
        """
        Implicit implementation.
        """
        acc = fwddyn_fn(x, u)
        _D = D(mpar, x)
        M = mass_matrix(mpar, x)
        A = jnp.linalg.inv(M - dt * _D)
        v = x[2:] + dt * A @ M @ acc
        q = x[:2] + dt * v
        new_x = jnp.hstack([q, v])
        return new_x

    return _implicit


def create_implicitfast_integrator(mpar, fwddyn_fn):
    """
    Wrapper to formulate Implicit Euler integrator.
    """

    def _implicit(dt, x, u):
        """
        Implicit implementation.
        """
        acc = fwddyn_fn(x, u)
        _D = D_fast(mpar, x)
        # D = jax.jacobian(combine_forces_fast, 2)(mpar, x[:2], x[2:], u)
        M = mass_matrix(mpar, x)
        A = jnp.linalg.inv(M - dt * _D)
        v = x[2:] + dt * A @ M @ acc
        q = x[:2] + dt * v
        new_x = jnp.hstack([q, v])
        return new_x

    return _implicit


def create_variational_integrator(mpar, converge=False):
    """
    Wrapper to formulate variational integrator based on Discrete
    Euler-Lagrange in Momentum Form. Default behaviour is to compute
    one-step stabilization of the Explicit Euler solution. Parameter
    `converge` allows to require the convergence of the method to
    tolerable solution.
    Args:
        mpar (model_parameters): Model parameters of the system
        dt (float): Solution timestep
        converge (bool): Whether required to converge
    Returns:
        Callable[jnp.ndarray, jnp.ndarray]: Step function, which computes the next state
                                            based on current state and control signal
    """
    # TODO: if convergence is planned to be used, then tolerance should be tunable

    # Build mass matrix based on given parameters
    __mass_matrix = partial(mass_matrix, mpar)
    # Build discrete lagrangian based on given parameters
    __discrete_lagrangian = partial(discrete_lagrangian, mpar)
    # Build discrete forces estimator based on given parameters
    __compute_discrete_force = partial(compute_discrete_force, mpar)

    def rf_cond(carry):
        """
        Rootfinding conditional.
        1) q2 advancement between iterations is above tolerable
        2) momentum error is above tolerable
        3) number of steps exceeds maximum
        All of three should satisfy to return True

        Args:
            carry tuple[]: configuration, momentum, configuration guess, previous
                            configuration guess, control signal, estimated momentum,
                            number of solver steps, solution tolerance
        Returns:
            bool: Whether to continue solution
        """
        _, p1, q2, prev_q2, _, p1_res, nsteps, atol, _ = carry
        # 1) q2 advancement between iterations is above tolerable
        q2_converged = jnp.linalg.norm(q2 - prev_q2) > atol
        # 2) momentum error is above tolerable
        p1_converged = jnp.linalg.norm(p1 + p1_res) > atol
        # 3) number of steps exceeds maximum
        nsteps_converged = nsteps < 3
        return q2_converged & p1_converged & nsteps_converged

    def rf_body(carry):
        """
        Rootfinding scan body. Computes the change in configuration, which
        reduces the desperancy in the observed and estimated momentum.

        Args:
            carry tuple[]: configuration, momentum, configuration guess, previous
                            configuration guess, control signal, estimated momentum,
                            number of solver steps, solution tolerance
        Returns:
            tuple[]: The same carry
        """
        q1, p1, q2, _, tau, _, nsteps, atol, dt = carry
        A = jax.jacobian(jax.grad(__discrete_lagrangian, 0), 1)(q1, q2, dt)
        F_l, _ = __compute_discrete_force(q1, q2, tau, dt)
        p1_res = jax.grad(__discrete_lagrangian, 0)(q1, q2, dt) + F_l
        gradient = jnp.linalg.inv(A) @ (p1_res + p1)
        return q1, p1, q2 - gradient, q2, tau, p1_res, nsteps + 1, atol, dt

    def rootfinder(dt, x, tau):
        """
        Variational Integrator based on Discrete Euler-Lagrange in Momentum Form.
        As initial guess on the configuration the explicit Euler is used. Then
        the solution is stabilized using rootfinding mechanism, based on the DELM
        equations. Finally, the system velocity is extracted from the resulting momentum.

        Args:
            x (jnp.ndarray): Current state of the system
            tau (jnp.ndarray): Control signal
        Returns:
            jnp.ndarray: The new state
        """
        M1 = __mass_matrix(x)
        q1 = x[:2]
        v1 = x[2:]
        # Compute ground truth momentum
        p1 = M1 @ v1

        q2 = q1 + v1 * dt

        # Estimate momentum based on configuration guess
        F_l, _ = __compute_discrete_force(q1, q2, tau, dt)
        p1_res = jax.grad(__discrete_lagrangian, 0)(q1, q2, dt) + F_l

        # Stabilize the solution
        if converge:
            _, _, q2, _, _, _, _, _, _ = jax.lax.while_loop(
                rf_cond, rf_body, (q1, p1, q2, q2 + 1, tau, p1_res, 0, 1e-3, dt)
            )
        else:
            _, _, q2, _, _, _, _, _, _ = rf_body((q1, p1, q2, q2 + 1, tau, p1_res, 0, 1e-3, dt))

        # Extract velocity from momentum
        M2 = __mass_matrix(q2)
        M2inv = jnp.linalg.inv(M2)
        _, F_r = __compute_discrete_force(q1, q2, tau, dt)
        p2 = jax.grad(__discrete_lagrangian, 1)(q1, q2, dt) + F_r
        v2 = M2inv @ p2

        return jnp.hstack([q2, v2])

    return rootfinder
