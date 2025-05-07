import jax
import jax.numpy as jnp

from parameters import mpar_nolim, goal
from double_pendulum.controller.vimppi.config import Config
from double_pendulum.controller.vimppi.controller import MPPIController
from copy import deepcopy

name = "vimppi"
leaderboard_config = {
    "csv_path": "trajectory.csv",
    "name": name,
    "simple_name": "vimppi",
    "short_description": "This controller does vimppi.",
    "readme_path": f"readmes/{name}.md",
    "username": "adk",
}

# torque_limit = mpar.tl
# torque_limit = [5.0, 0.5]
torque_limit = [0.5, 5.0]
mpar = deepcopy(mpar_nolim)
mpar.set_torque_limit(torque_limit)
dt = 0.005
t_final = 60.0
robot = "double_pendulum"

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
    baseline_control_type="zero",
    model_path="",
    robot="acrobot",
    lqr_dt=0.005,
    # For pendubot
    # sigma=jnp.diag(jnp.array([0.075, 0.075])),
    # For acrobot
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
