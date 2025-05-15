import os
import numpy as np

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

# model parameters
design = "design_C.1"
robot = "pendubot"
torque_limit = [5.0, 0.5]
torque_limit_con = [5.0, 0.0]
friction_compensation = True

model_par_path = "../../data/system_identification/identified_parameters/design_C.1/model_1.0/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
# mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit_con)

## trajectory parameters
csv_path = "../../data/trajectories/design_C.1/model_1.1/pendubot/ilqr_1/trajectory.csv"
dt = 0.002
t_final = 10.0

# swingup parameters
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.1

## controller parameters
lqr_path = "../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr"
lqr_pars = np.loadtxt(os.path.join(lqr_path, "controller_par.csv"))
Q_lqr = np.diag(lqr_pars[:4])
R_lqr = 8.0*np.diag([lqr_pars[4], lqr_pars[4]])

S = np.loadtxt(os.path.join(lqr_path, "Smatrix"))
rho = np.loadtxt(os.path.join(lqr_path, "rho"))

Q = np.diag([1.0, 1.0, 1.0, 1.0])
R = np.eye(2)*2.0
Qf = np.loadtxt(os.path.join(lqr_path, "Smatrix"))


# switiching conditions
def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]

    delta = wrap_angles_diff(np.subtract(x, goal))
    # print(x, delta)

    switch = False
    if np.einsum("i,ij,j", delta, S, delta) < 1.0 * rho:
        switch = True
        print(f"Switch to LQR at time={t}")

    return switch


# filter
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

# controller
controller1 = TVLQRController(
    model_pars=mpar_con,
    csv_path=csv_path,
)

controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller2 = LQRController(model_pars=mpar_con)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=100000)

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
)

controller.set_filter(filter)

if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
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
    motor_ids=[3, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, robot, "tvlqr"),
    record_video=True,
    safety_velocity_limit=30.0,
    #perturbation_array=perturbation_array,
)
