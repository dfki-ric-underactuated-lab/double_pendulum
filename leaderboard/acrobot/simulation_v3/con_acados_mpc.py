from datetime import datetime
import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.acados_mpc.acados_mpc import AcadosMpc

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

name="acados_mpc"
username="maranderine"

leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                      "name": name,
                      "simple_name": "AcadosMpc",
                      "short_description": "Real-Time nonlinear Model Predictive Conntrol implemented with Acados framework",
                      "readme_path": f"readmes/{name}.md",
                      "username": username
}
actuated_joint = np.argmax(mpar.tl)

# controller parameters
N_horizon=20
prediction_horizon=0.5
Nlp_max_iter=40
vmax = 20 #rad/s
vf = 20
bend_the_rules=True

if bend_the_rules:
    tl = mpar.tl
    tl[np.argmin(tl)] = 0.5
    mpar.set_torque_limit(tl)

if actuated_joint == 1: #acrobot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([100000, 100000, 1000, 1000])
    R_mat = 2*np.diag([0.000001, 0.000001])

if actuated_joint == 0: #pendubot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([100000, 100000, 1000, 1000]) 
    R_mat = 2*np.diag([0.000001, 0.000001])

controller = AcadosMpc(
    model_pars=mpar,
)

controller.set_start(x0)
controller.set_goal(goal)
controller.set_parameters(
    N_horizon=N_horizon,
    prediction_horizon=prediction_horizon,
    Nlp_max_iter=Nlp_max_iter,
    max_solve_time=.01,
    solver_type="SQP_RTI",
    wrap_angle=False,
    warm_start=True,
    fallback_on_solver_fail=True,
    nonuniform_grid=False,
    cheating_on_inactive_joint=bend_the_rules,
    mpc_cycle_dt=0.002,
    outer_cycle_dt=dt,
    qp_solver_tolerance = 0.01,
    qp_solver = 'PARTIAL_CONDENSING_HPIPM',
    hpipm_mode = 'ROBUST',
    vel_penalty=100000000
)

controller.set_velocity_constraints(v_max=vmax, v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)
#controller.load_init_traj(csv_path=init_csv_path)
controller.init()
