import os
import numpy as np

from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries

from sim_parameters import mpar, goal, x0, dt, integrator, design, robot

name = "ilqrfree"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "iLQR MPC",
    "short_description": "Online optimization with iterative LQR. Without reference trajectory.",
    "readme_path": f"readmes/{name}.md",
    "username": "fwiebe",
}

# # model parameters
torque_limit = [0.0, 6.0]

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# controller parameters
N = 400
N_init = 400
max_iter = 5
max_iter_init = 100
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.0001
break_cost_redu = 1e-6
trajectory_stabilization = False

f_sCu = [0.1, 0.1]
f_sCp = [10.0, 0.1]
f_sCv = [0.05, 0.2]
f_sCen = 0.0
f_fCp = [50.0, 10.0]
f_fCv = [1.0, 1.0]
f_fCen = 1.0

# construct simulation objects
controller = ILQRMPCCPPController(model_pars=mpar)
controller.set_start(x0)
controller.set_goal(goal)
controller.set_parameters(
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
controller.set_cost_parameters(
    sCu=f_sCu, sCp=f_sCp, sCv=f_sCv, sCen=f_sCen, fCp=f_fCp, fCv=f_fCv, fCen=f_fCen
)
controller.compute_init_traj(
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

controller.init()
