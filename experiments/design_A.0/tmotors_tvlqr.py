import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties
from double_pendulum.filter.lowpass import lowpass_filter

design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "pendubot"

torque_limit = [5.0, 5.0]
friction_compensation = True
stabilization = "lqr"

# model parameters
if robot == "pendubot":
    torque_limit_con = [5.0, 0.0]
elif robot == "acrobot":
    torque_limit_con = [0.0, 5.0]

model_par_path = (
    "../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
# mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit_con)

## trajectory parameters
csv_path = os.path.join(
    "../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

# T, X, U = load_trajectory(csv_path, True)
# dt, t_final, _, _ = trajectory_properties(T, X)
dt = 0.0025
t_final = 20.0

# swingup parameters
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.1

# controller parameters
if robot == "acrobot":
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = np.eye(2) * 0.82
elif robot == "pendubot":
    Q = 1.25 * np.diag([0.64, 0.56, 0.15, 0.14])
    # Q = 3.0*np.diag([0.64, 0.64, 0.1, 0.1])
    R = np.eye(2) * 0.82
    # R = np.eye(2)*1.5
Qf = np.copy(Q)
integrator = "runge_kutta"

## PID controller
Kp = 10.0
Ki = 0.0
Kd = 0.1

## lqr controller
if robot == "acrobot":
    Q_lqr = 0.1 * np.diag([0.65, 0.00125, 93.6, 0.000688])
    R_lqr = 100.0 * np.diag((0.025, 0.025))
elif robot == "pendubot":
    Q_lqr = np.diag([0.0125, 6.5, 6.88, 9.36])
    R_lqr = np.diag([0.024, 0.024])

## ilqr mpc controller
N = 100
con_dt = dt
N_init = 100
max_iter = 5
max_iter_init = 1000
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = False
shifting = 1
sCu = [0.0001, 0.0001]
sCp = [0.1, 0.1]
sCv = [0.01, 0.01]
sCen = 0.0
fCp = [10.0, 10.0]
fCv = [1.0, 1.0]
fCen = 0.0


# switiching conditions
def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]
    eps = [0.25, 0.25, 1.5, 1.5]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.0:
        print(f"Stayed with TVLQR control in state x {x} at time {t}")
        return False
    else:
        print(f"Switched to Stabilizing control in state x {x} at time {t}")
        return True


# filter
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

# controller
controller1 = TVLQRController(
    model_pars=mpar_con,
    csv_path=csv_path,
    num_break=40,
)

controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)

if stabilization == "pid":
    controller2 = PointPIDController(torque_limit=torque_limit, dt=dt)
    controller2.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
    controller2.set_goal(goal)
elif stabilization == "lqr":
    controller2 = LQRController(model_pars=mpar_con)
    controller2.set_goal(goal)
    controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
    controller2.set_parameters(failure_value=0.0, cost_to_go_cut=100000)

elif stabilization == "ilqr":
    controller2 = ILQRMPCCPPController(model_pars=mpar_con)
    controller2.set_goal(goal)
    controller2.set_parameters(
        N=N,
        dt=con_dt,
        max_iter=max_iter,
        regu_init=regu_init,
        max_regu=max_regu,
        min_regu=min_regu,
        break_cost_redu=break_cost_redu,
        integrator=integrator,
        trajectory_stabilization=trajectory_stabilization,
        shifting=shifting,
    )
    controller2.set_cost_parameters(
        sCu=sCu, sCp=sCp, sCv=sCv, sCen=sCen, fCp=fCp, fCv=fCv, fCen=fCen
    )
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

# run experiment
run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 8],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, robot, "tmotors/tvlqr_stab_results"),
)
