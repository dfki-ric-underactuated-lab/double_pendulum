import os
import numpy as np
import jax
import jax.numpy as jnp
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.utils.wrap_angles import wrap_angles_top, wrap_angles_diff
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.simulation.perturbations import (
    get_random_gauss_perturbation_array,
)
from double_pendulum.controller.vimppi.config import Config
from double_pendulum.controller.vimppi.controller import MPPIController
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
# model parameters
design = "design_C.1"
robot = "acrobot"
torque_limit = [0.5, 5.0]
torque_limit_con = [0.0, 5.0]
friction_compensation = False

dt = 0.002
t_final = 10.0

# swingup parameters
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.1


model_par_path = "../../data/system_identification/identified_parameters/design_C.1/model_1.0/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
# mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit_con)



torque_limit = mpar.tl

cfg = Config(
    key=jax.random.PRNGKey(0),
    horizon=15,
    samples=1024,
    exploration=0,
    lambda_=50.0,
    alpha=1.0,  # balance between low energy and smoothness
    # Disturbance detection parameters
    deviation_type="time",
    dx_delta_max=1e-1,
    dt_delta_max=0.02,
    # Baseline control parameters
    baseline_control_type="ar_eapo",
    model_path="../../data/policies/design_C.1/model_1.1/pendubot/AR_EAPO/model.zip",
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


# np.random.seed(2)
perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
    t_final, dt, 2, 1.0, [0.05, 0.1], [0.4, 0.6]
)

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[2, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, robot, "vimppi"),
    record_video=True,
    safety_velocity_limit=20.0,
    #perturbation_array=perturbation_array,
)
