import os
import numpy as np

from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_top

from sim_parameters import mpar, goal, x0, dt, integrator, design, robot

name = "ilqr_free_lqr"
leaderboard_config = {
    "csv_path": "data/" + name + "/sim_swingup.csv",
    "name": name,
    "username": "fwiebe",
}

# # model parameters
torque_limit = [0.0, 6.0]

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# controller parameters
N = 200
N_init = 2500
max_iter = 2
max_iter_init = 100
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.0001
break_cost_redu = 1e-6
trajectory_stabilization = False

f_sCu = [0.0001, 0.0001]
f_sCp = [10.0, 0.1]
f_sCv = [0.001, 0.2]
f_sCen = 0.0
f_fCp = [50.0, 10.0]
f_fCv = [1.0, 1.0]
f_fCen = 1.0

# LQR controller
Q = np.diag((0.97, 0.93, 0.39, 0.26))
R = np.diag((0.11, 0.11))


def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]
    eps = [0.2, 0.2, 1.0, 1.0]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.0:
        return False
    else:
        return True


# construct simulation objects
controller1 = ILQRMPCCPPController(model_pars=mpar)
controller1.set_start(x0)
controller1.set_goal(goal)
controller1.set_parameters(
    N=N,
    dt=dt,
    max_iter=max_iter,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    integrator=integrator,
    trajectory_stabilization=trajectory_stabilization,
)
controller1.set_cost_parameters(
    sCu=f_sCu, sCp=f_sCp, sCv=f_sCv, sCen=f_sCen, fCp=f_fCp, fCv=f_fCv, fCen=f_fCen
)
controller1.compute_init_traj(
    N=N_init,
    dt=dt,
    max_iter=max_iter_init,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    sCu=f_sCu,
    sCp=f_sCp,
    sCv=f_sCv,
    sCen=f_sCen,
    fCp=f_fCp,
    fCv=f_fCv,
    fCen=f_fCen,
    integrator=integrator,
)

# T, X, U = controller.get_init_trajectory()
# plot_timeseries(T, X, U)
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=1000)
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
)


controller.init()
