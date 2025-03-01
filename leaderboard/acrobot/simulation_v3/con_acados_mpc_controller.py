from datetime import datetime
import numpy as np
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.acados_mpc.acados_mpc_controller import AcadosMpcController

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

name="acados_mpc_controller"
username="maranderine"

leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                      "name": name,
                      "simple_name": "simple name",
                      "short_description": "Short controller description (max 100 characters)",
                      "readme_path": f"readmes/{name}.md",
                      "username": username
}
bend_the_rules=True

if bend_the_rules:
    tl = mpar.tl
    tl[np.argmin(tl)] = 0.5
    mpar.set_torque_limit(tl)

actuated_joint = np.argmax(tl)

# controller parameters
N_horizon=20
prediction_horizon=2.0
Nlp_max_iter=200
vmax = 30 #rad/s
vf = 0

if actuated_joint == 1: #acrobot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([10000, 10000, 100, 100])
    R_mat = 2*np.diag([0.0001, 0.0001])

if actuated_joint == 0: #pendubot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([10000, 10000, 100, 100]) 
    R_mat = 2*np.diag([0.0001, 0.0001])

controller = AcadosMpcController(
    model_pars=mpar,
    cheating_on_inactive_joint= np.min(tl)
)

controller.set_start(x0)
controller.set_goal(goal)
controller.set_parameters(
    N_horizon=N_horizon,
    prediction_horizon=prediction_horizon,
    Nlp_max_iter=Nlp_max_iter,
    solver_type="SQP",
    wrap_angle=False
)

#controller.set_velocity_constraints(v_max=vmax)
#controller.set_velocity_constraints(v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)
#controller.load_init_traj(csv_path=init_csv_path)
controller.init()

"""
friction compensation: torque on the passive joint must not exceed 0.5 Nm
Control loop frequency: 700 Hz maximum
Torque limit: 6 Nm
Velocity limit: 30 rad/s
Position limits: ± 540 degrees for both joints

rules:
---------------
Each attempt has an attempt of a total time duration of 60s (swing ups + stabilizations).
During the 60s run, at a maximum of 15 random times the controller will be deactivated and the system will be reset do a random state (max duration 1s).

todos:
---------
-maybe model path restriction
- run leaderboard code
-start paper
"""
