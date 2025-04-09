"""
Dynamics module. Defines all the necessary dynamics related functions.
"""

import jax.numpy as jnp
from double_pendulum.model.model_parameters import model_parameters


def mass_matrix(mpar, x):
    """
    Compute inertia matrix of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Mass matrix
    """
    pos = x[0:2]
    m00 = (
        mpar.I[0]
        + mpar.I[1]
        + mpar.m[1] * mpar.l[0] ** 2.0
        + 2 * mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.cos(pos[1])
        + mpar.gr**2.0 * mpar.Ir
        + mpar.Ir
    )
    m01 = mpar.I[1] + mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.cos(pos[1]) - mpar.gr * mpar.Ir
    m10 = mpar.I[1] + mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.cos(pos[1]) - mpar.gr * mpar.Ir
    m11 = mpar.I[1] + mpar.gr**2.0 * mpar.Ir

    return jnp.array([[m00, m01], [m10, m11]])


def coriolis(mpar, x):
    """
    Compute Coriolis term of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Coriolis term
    """
    pos = x[0:2]
    vel = x[2:4]

    c00 = -2 * mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[1]
    c01 = -mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[1]
    c10 = mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[0]
    c11 = 0

    return jnp.array([[c00, c01], [c10, c11]])


def dCvdv(mpar, x):
    """
    Analytical derivative of coriolis forces [C(v) @ v] with respect to v.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: d(Cv)dv
    """
    pos = x[0:2]
    vel = x[2:4]

    _dCvdv = jnp.array(
        [
            [
                -2 * mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[1],
                -2 * mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[0]
                + -2 * mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[1],
            ],
            [
                2 * mpar.m[1] * mpar.l[0] * mpar.r[1] * jnp.sin(pos[1]) * vel[0],
                0,
            ],
        ]
    )

    return _dCvdv


def gravity(mpar, x):
    """
    Compute gravity term of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Gravity term
    """
    pos = x[0:2]
    g0 = -mpar.m[0] * mpar.g * mpar.r[0] * jnp.sin(pos[0]) - mpar.m[1] * mpar.g * (
        mpar.l[0] * jnp.sin(pos[0]) + mpar.r[1] * jnp.sin(pos[0] + pos[1])
    )
    g1 = -mpar.m[1] * mpar.g * mpar.r[1] * jnp.sin(pos[0] + pos[1])
    return jnp.array([g0, g1])


def friction(mpar, x):
    """
    Compute friction term of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Friction term
    """
    vel = x[2:4]

    f0 = jnp.array(
        [
            mpar.b[0] * vel[0] + mpar.cf[0] * jnp.arctan(100 * vel[0]),
            mpar.b[1] * vel[1] + mpar.cf[1] * jnp.arctan(100 * vel[1]),
        ]
    )

    return f0


def dfdv(mpar, x):
    """
    Analytical derivative of the system friction w.r.t. velocity.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: dfdv
    """
    vel = x[2:4]
    _dfdv = jnp.array(
        [
            [mpar.b[0] + mpar.cf[0] * 100 / (10000 * vel[0] * vel[0] + 1), 0],
            [0, mpar.b[1] + mpar.cf[1] * 100 / (10000 * vel[1] * vel[1] + 1)],
        ]
    )
    return _dfdv


def kinetic_energy(mpar, x):
    """
    Compute kinetic energy of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Kinetic energy
    """
    vel = x[2:]
    M = mass_matrix(mpar, x)
    return 0.5 * vel.T @ M @ vel


def potential_energy(mpar, x):
    """
    Compute potential energy of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Potential energy
    """
    pos = x[:2]
    y1 = -mpar.r[0] * jnp.cos(pos[0])
    y2 = -mpar.l[0] * jnp.cos(pos[0]) - mpar.r[1] * jnp.cos(pos[1] + pos[0])
    return mpar.m[0] * mpar.g * y1 + mpar.m[1] * mpar.g * y2


def lagrangian(mpar, x):
    """
    Compute Lagrangian of the system.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
    Returns:
        jnp.ndarray: Lagrangian
    """
    return kinetic_energy(mpar, x) - potential_energy(mpar, x)


def quadrature_rule(q0, q1, dt):
    """
    Trapezoidal quadrature rule.

    Args:
        q1 (jnp.ndarray): First configuration
        q2 (jnp.ndarray): Second configuration
        dt: (float): Timestep
    Returns:
        jnp.ndarray, jnp.ndarray: Midpoint configuration and velocity
    """
    return (q1 + q0) / 2, (q1 - q0) / dt


def discrete_lagrangian(mpar, q0, q1, dt):
    """
    Discretization of the Lagrangian
    Args:
        mpar (model_parameters): Model parameters
        q0 (jnp.ndarray): First configuration
        q1 (jnp.ndarray): Second configuration
        dt (float): Timestep
    Returns:
        jnp.ndarray: Lagrangian
    """
    q, dq = quadrature_rule(q0, q1, dt)
    new_state = jnp.hstack([q, dq])
    return lagrangian(mpar, new_state) * dt


def force(mpar, x, tau):
    """
    Total force acting on the system
    (excluding potential forces).

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
        tau (jnp.ndarray): Control signal
    Returns:
        jnp.ndarray: Resulting force
    """
    return tau - friction(mpar, x)


def discrete_force(mpar, x, tau, dt):
    """
    Discretization of the resulting force in the system.
    Splits the force in two discrete forces.

    Left) 0.5 * F * dt
    Right) 0.5 * F * dt

    Discretization is good enough even though both forces
    are using the same state of the system. In original
    formulation left are right forces should use different
    state approximations.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): System state
        tau (jnp.ndarray): Control signal
        dt (float): Timestep
    Returns:
        jnp.ndarray, jnp.ndarray: Left and right discrete forces
    """
    return 0.5 * force(mpar, x, tau) * dt, 0.5 * force(mpar, x, tau) * dt


def compute_discrete_force(mpar, q1, q2, tau, dt):
    """
    Estimates discrete force based on the two configurations.

    Args:
        mpar (model_parameters): Model parameters
        q1 (jnp.ndarray): First configuration
        q2 (jnp.ndarray): Second configuration
        tau (jnp.ndarray): Control signal
        dt (float): Timestep
    Returns:
        jnp.ndarray, jnp.ndarray: Left and right discrete forces in midpoint state
    """
    q, dq = quadrature_rule(q1, q2, dt)
    return discrete_force(mpar, jnp.hstack([q, dq]), tau, dt)


def create_dynamics(mpar: model_parameters):
    """
    Create forward dynamics based on model parameters

    Args:
        mpar (model_parameters): Model parameters
    Returns:
        Callable[jnp.ndarray, jnp.ndarray]: Forward dynamics function
    """
    # compute B matrix given the torque limit
    if mpar.tl[0] == 0.0:
        B = jnp.array([[0.0, 0.0], [0.0, 1.0]])
    elif mpar.tl[1] == 0.0:
        B = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    else:
        B = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    def forward_dynamics(x, tau):
        """
        Analytical form of forward dynamics.

        Args:
            x (jnp.ndarray): Current state
            tau (jnp.ndarray): Control signal
        Returns:
            jnp.ndarray: Estimated acceleration
        """
        vel = x[2:4]

        M = mass_matrix(mpar, x)
        C = coriolis(mpar, x)
        G = gravity(mpar, x)
        F = friction(mpar, x)

        Minv = jnp.linalg.inv(M)
        force = G - C @ vel + B @ tau
        return Minv @ (force - F)

    return forward_dynamics


def D(mpar, x):
    """
    Analytical form of the matrix D from velocity-implicit method.
    da/dv = M_inv @ D

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): Current robot state
        tau (jnp.ndarray): Control signal
    Returns:
        jnp.ndarray: matrix D
    """
    return -dfdv(mpar, x) - dCvdv(mpar, x)


def D_fast(mpar, x):
    """
    Analytical form of the matrix D from fast velocity-implicit method.
    da/dv = M_inv @ D
    Fast implementation does not compute coriolis term.

    Args:
        mpar (model_parameters): Model parameters
        x (jnp.ndarray): Current robot state
        tau (jnp.ndarray): Control signal
    Returns:
        jnp.ndarray: Matrix D
    """
    return -dfdv(mpar, x)
