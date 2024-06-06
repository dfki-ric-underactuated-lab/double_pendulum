import os
from datetime import datetime
import numpy as np
import yaml

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_top

from sim_parameters import (
    mpar,
    dt,
    t_final,
    t0,
    x0,
    goal,
    integrator,
    design,
    model,
    robot,
)

name = "ilqr_ilqrmpc_lqr"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "iLQR MPC stabilization",
    "short_description": "Online optimization with iterative LQR. Stabilization of iLQR trajectory. Top stabilization with LQR.",
    "readme_path": f"readmes/{name}.md",
    "username": "fwiebe",
}

traj_model = "model_1.1"

# controller parameters
# N = 20
N = 100
con_dt = dt
N_init = 1000
max_iter = 10
# max_iter = 100
max_iter_init = 1000
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = True
shifting = 1

init_csv_path = os.path.join(
    "../../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

sCu = [0.1, 0.1]
sCp = [0.1, 0.1]
sCv = [0.01, 0.1]
sCen = 0.0
fCp = [100.0, 10.0]
fCv = [10.0, 1.0]
fCen = 0.0

f_sCu = [0.1, 0.1]
f_sCp = [0.1, 0.1]
f_sCv = [0.01, 0.01]
f_sCen = 0.0
f_fCp = [10.0, 10.0]
f_fCv = [1.0, 1.0]
f_fCen = 0.0

# sCu = [0.1, 0.1]
# sCp = [0.1, 0.1]
# sCv = [0.01, 0.1]
# sCen = 0.0
# fCp = [100.0, 10.0]
# fCv = [10.0, 1.0]
# fCen = 0.0
#
# f_sCu = [0.0001, 0.0001]
# f_sCp = [1.0, 1.0]
# f_sCv = [0.1, 0.1]
# f_sCen = 0.0
# f_fCp = [10.0, 10.0]
# f_fCv = [1.0, 1.0]
# f_fCen = 0.0

# LQR controller
Q = np.diag((0.97, 0.93, 0.39, 0.26))
R = np.diag((0.11, 0.11))


def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]
    eps = [0.14, 0.14, 0.5, 0.5]
    # eps = [0.2, 0.2, 1.0, 1.0]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.0:
        return False
    else:
        return True


controller1 = ILQRMPCCPPController(model_pars=mpar)
controller1.set_start(x0)
controller1.set_goal(goal)
controller1.set_parameters(
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
controller1.set_cost_parameters(
    sCu=sCu, sCp=sCp, sCv=sCv, sCen=sCen, fCp=fCp, fCv=fCv, fCen=fCen
)
controller1.set_final_cost_parameters(
    sCu=f_sCu, sCp=f_sCp, sCv=f_sCv, sCen=f_sCen, fCp=f_fCp, fCv=f_fCv, fCen=f_fCen
)
controller1.load_init_traj(csv_path=init_csv_path, num_break=40, poly_degree=3)

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
