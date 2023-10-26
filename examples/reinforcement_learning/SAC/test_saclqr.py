import os
from datetime import datetime

import matplotlib.pyplot
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_top,wrap_angles_diff

from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)

# hyperparameters
stabilization = "lqr"
# robot = "pendubot"
robot = "acrobot"

if robot == "pendubot":
    ## case design_A.0 model_2.0
    # design = "design_A.0"
    # model = "model_2.0"
    # load_path = ""

    ## case design_C.1 model_1.0
    design = "design_C.1"
    model = "model_1.0"
    load_path = ("../../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr/")
    scaling_state = False
    torque_limit = [5.0, 0.0]
    active_act = 0

    ## gym version
    # design A.0
    # model_path = "../../../data/policies/design_A.0/model_2.0/pendubot/SAC/sac_model"
    ## gymnasium versionq
    # design C.1
    # model_path = "../../../data/policies/design_C.1/model_1.0/pendubot/SAC/best_model.zip"
    model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/train_sac_without_limit/saved_model/pendubot/working_2e7_new/best_model.zip"
elif robot == "acrobot":
    ## case design_C.0 model_3.0
    # design = "design_C.0"
    # model = "model_3.0"
    # scaling_state = False
    # load_path = ""

    ## case design_C.1 model_1.0
    design = "design_C.1"
    model = "model_1.0"
    load_path = ("../../../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr/")
    scaling_state = True

    torque_limit = [0.0, 5.0]
    active_act = 1

    # gym version
    # model_path = "../../../data/policies/design_C.0/model_3.0/acrobot/SAC/sac_model"
    # gymnasirum version
    # model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/log_data/SAC_training/saved_models/acrobot/design_C.0/model_3.0/gymnasium_3e7/best_model.zip"

    # for testing
    # model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/saved_model/acrobot/actually_works/best_model.zip"
    model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/log_data_designC.1/SAC_training/best_model/best_model.zip"
    # model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/saved_model/acrobot/another_working/best_model.zip"
    # model_path = "/home/chi/backup/double_pendulum_aug/examples/reinforcement_learning/SAC/saved_models/acrobot/design_C.1/model_1.0/works_very_well/best_model.zip"
    # model_path = "/home/chi/backup/double_pendulum_aug/examples/reinforcement_learning/SAC/saved_models/acrobot/design_C.1/model_1.0/working_speed_limit_15/best_model.zip"
# import model parameter
model_par_path = (
        "../../../data/system_identification/identified_parameters/"
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
dt = 0.0025
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

# switching conditions
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))
flag = False

# LQR parameters
lqr_pars = np.loadtxt(os.path.join(load_path, "controller_par.csv"))
Q = np.diag(lqr_pars[:4])
R = np.diag([lqr_pars[4], lqr_pars[4]])

def condition1(t, x):
    return False

def check_if_state_in_roa(S, rho, x):
    # print(x)
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    print(rad, rho)
    return rad < 1.0*rho, rad

def condition2(t, x):
    # print("x=",x)
    # y = wrap_angles_top(x)
    y = wrap_angles_diff(x)
    # print("y=",y)
    flag,rad = check_if_state_in_roa(S,rho,y)
    print(rad,rho)
    if flag:
        print(t)
        print(y)
        print(flag)
        return flag
    return flag

# def condition2(t,x):
#     return False

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    scaling = scaling_state
)

# initialize sac controller
controller1 = SACController(
    model_path = model_path,
    dynamics_func=dynamics_func,
    dt=dt,
    scaling = scaling_state
)

# initialize lqr controller
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1000)

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    # save_video=False,
)

# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    # save_to="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/plots/acrobot.png"
)
