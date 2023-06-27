import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

from double_pendulum.controller.SAC.SAC_controller import SACController
from src.python.double_pendulum.simulation.gym_env8 import (
    double_pendulum_dynamics_func,
)
# from src.python.double_pendulum.controller.SAC.check_state import check_if_state_in_roa
from double_pendulum.utils.wrap_angles import wrap_angles_diff

## model parameters
# pendubot
design = "design_A.0"
model = "model_2.0"
robot = "pendubot"

# acrobot
# design = "design_C.0"
# model = "model_3.0"
# robot = "acrobot"

friction_compensation = True
stabilization = "lqr"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1



model_par_path = (
        "/home/chi/Github/double_pendulum/data/system_identification/identified_parameters/"
        + design
        + "/"
        + model
        + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)


# simulation parameters
dt = 0.01
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

def condition1(t, x):
    return False

# my version
if robot == "acrobot":
    Q = np.diag((0.0125, 6.5, 6.88, 9.36))
    R = np.diag((2.4, 2.4))
    load_path = "data/acrobot/lqr/roa"
    # Q = np.diag([1.0, 1.0, 1.0, 1.0])
    # R = np.diag((2.4, 2.4))

elif robot == "pendubot":
    Q = 3.0*np.diag([0.64, 0.64, 0.1, 0.1])
    R = np.eye(2)*0.82
    load_path = "data/pendubot/lqr/roa"

# v0 of switching condition
def condition2(t, x):
    goal = [np.pi, 0., 0., 0.]
    # pendubot
    eps = [0.3, 0.3, 1.2, 1.2]
    # acrobot
    # eps = [0.4, 0.4, 1.0, 1.0]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    print(max_diff)
    if max_diff > 1.0:
        return False
    else:
        print(t)
        print(y)
        print(max_diff)
        return True

#############################################################
# v1 of switching condition
# rho = np.loadtxt(os.path.join(load_path, "rho"))
# vol = np.loadtxt(os.path.join(load_path, "vol"))
# S = np.loadtxt(os.path.join(load_path, "Smatrix"))
# flag = False
#
# def check_if_state_in_roa(S, rho, x):
#     # print(x)
#     xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
#     rad = np.einsum("i,ij,j", xdiff, S, xdiff)
#     print(rad, rho)
#     return rad < 1.0*rho, rad
# def condition2(t, x):
#     # print("x=",x)
#     y = wrap_angles_top(x)
#     # y = wrap_angles_top()
#     # print("y=",y)
#     flag,rad = check_if_state_in_roa(S,rho,y)
#     print(rad,rho)
#     if flag:
#         print(t)
#         print(y)
#         print(flag)
#         return flag
#     return flag
######################################################

dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
)

controller1 = SACController(
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/best_model/acrobot_model.zip",
    # pendubot
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v0_5e6/acrobot_model.zip",
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v1_1e6/acrobot_model.zip",
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v2/acrobot_model.zip",
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v3/acrobot_model.zip",
    model_path ="/src/python/double_pendulum/controller/SAC/sac_training/pendubot/v4_3e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v5_final/acrobot_model.zip",

    # acrobot with quadratic reward
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v3_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v4_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v5_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v6_1e6/acrobot_model.zip",

    # acrobot with quadratic + bonus reward
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v2/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v4_with_termination/acrobot_model.zip",

    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_speed_modified/v1_5e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_speed_modified/v2_5e6_final/acrobot_model.zip",
    dynamics_func=dynamics_func,
    dt=dt,
)

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15)

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)
controller.init()

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
)