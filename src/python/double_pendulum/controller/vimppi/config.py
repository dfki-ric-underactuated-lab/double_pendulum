"""
Configuration module. Configuration serves parameters to the controller
and dynamics, as well as MPPI, formulates the costs and integrators.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import jax.typing as jpt
from double_pendulum.model.model_parameters import model_parameters

from .baseline_control import (
    create_tv_lqr_control_generator,
    create_zero_control_generator,
    create_ar_eapo_generator,
)
from .disturbance_checker import create_time_disturbance_checker, createt_state_disturbance_checker
from ...simulation.variational_integrator import (
    create_explicit_euler_integrator,
    create_implicit_integrator,
    create_implicitfast_integrator,
    create_variational_integrator,
)
from ...model.jax_plant import create_dynamics

UPWARD_GOAL = jnp.array([jnp.pi, 0.0, 0.0, 0.0])


@dataclass
class Config:
    key: jpt.ArrayLike  # random key

    # Rollout horizon
    horizon: int
    # Number of rollouts
    samples: int
    # Exploration length
    exploration: float
    # MPPI Hyperparameters
    # TODO: name them
    lambda_: float
    alpha: float
    sigma: jpt.ArrayLike

    state_dim: int
    act_dim: int
    act_min: jpt.ArrayLike
    act_max: jpt.ArrayLike

    # State weights
    Qdiag: jpt.ArrayLike
    # Action weights
    Rdiag: jpt.ArrayLike
    # Terminal state weights
    Pdiag: jpt.ArrayLike
    # Terminal cost coefficient
    terminal_coeff: float

    # Rollout timestep
    mppi_dt: float
    # Integrator selector
    mppi_integrator: str

    # Model parameters
    mpar: model_parameters

    # Disturbance detector Hyperparameters
    deviation_type: str
    dt_delta_max: float
    dx_delta_max: float

    # Baseline control generator hyperparameters
    baseline_control_type: str
    lqr_dt: float
    model_path: str
    robot: str

    # Dynamics model
    forward_dynamics: Callable[[jpt.ArrayLike, jpt.ArrayLike], jpt.ArrayLike] | None = None
    # Step function
    step: Callable[[jpt.ArrayLike, jpt.ArrayLike, float], jpt.ArrayLike] | None = None
    # Stage cost
    stage_cost: Callable[[jpt.ArrayLike, jpt.ArrayLike], jpt.ArrayLike] | None = None
    # Terminal cost
    terminal_cost: Callable[[jpt.ArrayLike], jpt.ArrayLike] | None = None
    # Disturbance detector
    check_disturbance: Callable[[jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike, jpt.ArrayLike], bool] | None = None
    # Baseline control generator
    baseline_control_generator: Callable[[jpt.ArrayLike], jpt.ArrayLike] | None = None

    def __post_init__(self):
        """
        Initialize internal configuration parameters
        """
        self.update_mpar(self.mpar)
        self.update_goal(UPWARD_GOAL)

    def initialize_dynamics(self):
        """
        Initialize system forward dynamics
        """
        self.forward_dynamics = create_dynamics(self.mpar)

    def update_mpar(self, mpar: model_parameters):
        """
        Update the configuration based on the new model parameters

        Args:
            mpar (model_parameters): Model parameters
        """

        # Set the model parameters
        self.mpar = mpar
        # Recalculate the dynamics
        self.initialize_dynamics()

        # Initialize integrator
        if self.mppi_integrator == "variational":
            self.step = create_variational_integrator(self.mpar)
        elif self.mppi_integrator == "explicit":
            self.step = create_explicit_euler_integrator(self.forward_dynamics)
        elif self.mppi_integrator == "implicit":
            self.step = create_implicit_integrator(self.mpar, self.forward_dynamics)
        elif self.mppi_integrator == "implicitfast":
            self.step = create_implicitfast_integrator(self.mpar, self.forward_dynamics)
        else:
            # TODO: Implement RK4
            raise ValueError(f"Unknown integrator type: {self.mppi_integrator}")

        # Initialize disturbance detector
        if self.deviation_type == "time":
            self.check_disturbance = create_time_disturbance_checker(self.dt_delta_max)
        elif self.deviation_type == "state":
            # FIXME: this would break if mppi_dt != None
            self.check_disturbance = createt_state_disturbance_checker(self.dx_delta_max, self.step)
        else:
            raise ValueError(f"Unknown disturbance detection type: {self.deviation_type}")

        # Initialize baseline control generator
        if self.baseline_control_type == "zero":
            self.baseline_control_generator = create_zero_control_generator(self.horizon, self.act_dim)
        elif self.baseline_control_type == "tv_lqr":
            if self.mppi_dt is None:
                raise ValueError("mppi_dt must be set for tv_lqr baseline control")
            B = jnp.zeros((self.state_dim, self.act_dim))
            if mpar.tl[0] == 0.0:
                B = B.at[3, 1].set(1.0)
            elif mpar.tl[1] == 0.0:
                B = B.at[2, 0].set(1.0)
            else:
                B = B.at[2, 0].set(1.0)
                B = B.at[3, 1].set(1.0)

            self.baseline_control_generator = create_tv_lqr_control_generator(
                self.forward_dynamics,
                lqr_dt=self.lqr_dt,
                mppi_horizon=self.horizon,
                mppi_dt=self.mppi_dt,
                nu=self.act_dim,
                B=B,
                Q=jnp.diag(self.Qdiag),
                R=jnp.diag(self.Rdiag),
                Q_final=jnp.diag(self.Pdiag),
            )
        elif self.baseline_control_type == "ar_eapo":
            self.baseline_control_generator = create_ar_eapo_generator(
                self.step,
                self.horizon,
                self.mppi_dt,
                model_path=self.model_path,
                robot=self.robot,
            )

    def update_goal(self, goal: jpt.ArrayLike):
        """
        Update the configuration goal and
        cost functions based on the goal.

        Args:
            goal (jnp.ndarray): Goal configuration
        """
        goal = jnp.array(goal)
        # we have to transform goal into global frame
        goal = goal.at[1].set(goal[0] + goal[1])

        Q = jnp.diag(self.Qdiag)
        R = jnp.diag(self.Rdiag)
        P = jnp.diag(self.Pdiag)

        def _state_error(x, target_x):
            """
            Compute state error.

            Args:
                x (jnp.ndarray): State
                target_x (jnp.ndarray): Target state
            Returns:
                jnp.ndarray: State error
            """
            # transform x to global frame
            x = x.at[1].set(x[0] + x[1])

            dx = x - target_x
            dx = dx.at[:2].set((x[:2] - target_x[:2]) % (2 * jnp.pi))
            dx = dx.at[:2].set(jnp.where(dx[:2] > jnp.pi, dx[:2] - 2 * jnp.pi, dx[:2]))

            return dx

        def stage_cost(x, u):
            """
            Compute running cost of current state and control signal.

            Args:
                x (jnp.ndarray): System state
                u (jnp.ndarray): Control signal
            Returns:
                float: Running cost
            """
            dx = _state_error(x, goal)
            return jnp.dot(jnp.dot(dx, Q), dx) + jnp.dot(jnp.dot(u, R), u)

        def terminal_cost(x):
            """
            Compute terminal cost of the state

            Args:
                x (jnp.ndarray): System state
            Returns:
                float: Terminal cost
            """
            dx = _state_error(x, goal)
            return jnp.dot(jnp.dot(dx, P), dx) * self.terminal_coeff

        self.stage_cost = stage_cost
        self.terminal_cost = terminal_cost

    def save_(self, save_dir):
        """
        Save Config parameters to a directory.

        Args:
            save_dir (str): directory to save the parameters.
        """
        import json

        def jax_to_serializable(obj):
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()  # Convert JAX array to list
            if isinstance(obj, model_parameters):
                return obj.__dict__
            if isinstance(obj, functools.partial):
                return "partial function: " + obj.func.__name__
            if isinstance(obj, Callable):
                return "not serializable function: " + obj.__name__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(self.__dict__, f, indent=4, default=jax_to_serializable)
