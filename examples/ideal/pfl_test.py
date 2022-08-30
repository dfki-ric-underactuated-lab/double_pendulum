import sys
import os
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import (SymbolicPFLController,
                                                                                    SymbolicPFLAndLQRController)

# model parameters
robot = "acrobot"
pfl_method = "collocated"

if robot == "acrobot":
    torque_limit = [0.0, 50.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [50.0, 0.0]
    active_act = 0

model_par_path = "../../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_gravity(0.)
mpar.set_torque_limit(torque_limit)

# simulation parameters
integrator = "runge_kutta"
goal = [np.pi/2, np.pi/4, 0., 0.]
dt = 0.01
x0 = [0.1, 0.0, 0.0, 0.0]
t_final = 10.0

# controller parameters
if robot == "acrobot":
    # lqr parameters
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
    if pfl_method == "collocated":
        par = [6.78389278, 5.66430937, 0.]
        #par = [1.58316202e-03, 2.94951787e+00, 1.44919303e+00]
        #par = [19.95373044, 14.76768604, 18.23010249]
        #par = [9.83825279, 9.42196979, 7.56036347]
    elif pfl_method == "noncollocated":
        par = [9.19534629, 2.24529733, 5.90567362]
elif robot == "pendubot":
    # lqr parameters
    Q = np.diag([0.00125, 0.65, 0.000688, 0.000936])
    R = np.diag([25.0, 25.0])
    if pfl_method == "collocated":
        par = [8.0722899, 4.92133648, 3.53211381]
    elif pfl_method == "noncollocated":
        #par = [26.34039456, 99.99876263, 11.89097532]
        par = [8.0722899, 4.92133648, 3.53211381]

plant = SymbolicDoublePendulum(model_pars=mpar)

controller = SymbolicPFLController(model_pars=mpar,
                                   robot=robot,
                                   pfl_method=pfl_method)

sim = Simulator(plant=plant)

controller.set_goal(goal)
controller.set_cost_parameters_(par)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=x0,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator=integrator,
                                   phase_plot=False,
                                   save_video=False)

plot_timeseries(T=T, X=X, U=U,
                pos_y_lines=[np.pi/2, np.pi/4])

