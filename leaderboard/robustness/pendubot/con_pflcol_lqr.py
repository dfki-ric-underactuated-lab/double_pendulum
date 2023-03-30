import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import (
    SymbolicPFLAndLQRController,
)
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

name = "pflcol_lqr"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "Energy PFL",
    "short_description": "Partial Feedback Linearization with energy shaping control. Stabilization with LQR.",
    "readme_path": f"readmes/{name}.md",
    "username": "fwiebe",
}

pfl_method = "collocated"

torque_limit = [3.0, 0.0]

mpar.set_torque_limit(torque_limit)

# controller parameters
Q = np.diag([0.00125, 0.65, 0.000688, 0.000936])
R = np.diag([25.0, 25.0])
par = [1.0, 1.0, 0.5]

controller = SymbolicPFLAndLQRController(
    model_pars=mpar, robot=robot, pfl_method=pfl_method
)
controller.lqr_controller.set_cost_parameters(
    p1p1_cost=Q[0, 0],
    p2p2_cost=Q[1, 1],
    v1v1_cost=Q[2, 2],
    v2v2_cost=Q[3, 3],
    p1v1_cost=0.0,
    p1v2_cost=0.0,
    p2v1_cost=0.0,
    p2v2_cost=0.0,
    u1u1_cost=R[0, 0],
    u2u2_cost=R[1, 1],
    u1u2_cost=0.0,
)

controller.set_goal(goal)
controller.set_cost_parameters_(par)
controller.init()
