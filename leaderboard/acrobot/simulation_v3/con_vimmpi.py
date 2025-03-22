import jax
import jax.numpy as jnp

from sim_parameters import (
    goal,
    mpar,
)

from double_pendulum.controller.vimppi.config import Config
from double_pendulum.controller.vimppi.controller import MPPIController

name = "vimppi"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "VIMPPI-AR-EAPO",
    "short_description": "Stabilization of iLQR trajectory with time-varying LQR.",
    "readme_path": f"readmes/{name}.md",
    "username": "adk",
}


torque_limit = mpar.tl

cfg = Config(
    key=jax.random.PRNGKey(0),
    horizon=20,
    samples=4096,
    exploration=0,
    lambda_=50.0,
    alpha=1.0,  # balance between low energy and smoothness
    # Disturbance detection parameters
    deviation_type="time",
    dx_delta_max=1e-1,
    dt_delta_max=0.02,
    # Baseline control parameters
    baseline_control_type="ar_eapo",
    model_path="../../../data/policies/design_C.1/model_1.1/acrobot/AR_EAPO/model.zip",
    robot="acrobot",
    lqr_dt=0.005,
    sigma=jnp.diag(jnp.array([0.2, 0.2])),
    state_dim=4,
    act_dim=2,
    act_min=-jnp.array(torque_limit),
    act_max=jnp.array(torque_limit),
    Qdiag=jnp.array([10.0, 1.0, 0.1, 0.1]),
    Rdiag=jnp.array([0.1, 0.1]),
    Pdiag=jnp.array([5.0, 5.0, 2.0, 2.0]),
    terminal_coeff=1e6,
    mppi_dt=0.02,
    mpar=mpar,
    mppi_integrator="variational",
)

controller = MPPIController(config=cfg)
controller.set_goal(goal)
controller.init()
