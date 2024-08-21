import os
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)
"""
This testing script is purely for testing the behaviour of SAC controller in swing-up task.
"""



# hyperparameters
# robot = "pendubot"
robot = "acrobot"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

    design = "design_A.0"
    model = "model_2.0"
    model_path = "../../../data/policies/design_A.0/model_2.0/pendubot/SAC/sac_model"
    scaling_state = True

    # design = "design_C.1"
    # model = "model_1.0"
    # model_path = "../../../data/policies/design_C.1/model_1.0/pendubot/SAC/sac_model.zip"
    # scaling_state = False

elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

    # design = "design_C.0"
    # model = "model_3.0"
    # model_path = "../../../data/policies/design_C.0/model_3.0/acrobot/SAC/sac_model.zip"
    # scaling_state = True

    design = "design_C.1"
    model = "model_1.0"
    scaling_state = True
    model_path = "../../../data/policies/design_C.1/model_1.0/acrobot/SAC/sac_model.zip"

# import model parameter
model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

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

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    scaling=scaling_state
)

# initialize sac controller
controller = SACController(
    model_path = model_path,
    dynamics_func=dynamics_func,
    dt=dt,
    scaling=scaling_state
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
    save_video=False,
)

# plot time series
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
)