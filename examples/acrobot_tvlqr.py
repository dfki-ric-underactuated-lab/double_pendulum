import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.utils.plotting import plot_timeseries


robot = "acrobot"
csv_path = "data/dircol/acrobot_swingup_1000Hz.csv"
urdf_path = "../data/urdfs/acrobot.urdf"

# model parameters
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 6.0]

# simulation parameters

trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
T_des = trajectory.T[0]
pos1_des = trajectory.T[1]
vel1_des = trajectory.T[2]
tau1_des = trajectory.T[5]
pos2_des = trajectory.T[6]
vel2_des = trajectory.T[7]
tau2_des = trajectory.T[10]

U_des = np.vstack((tau1_des, tau2_des)).T
X_des = np.vstack((pos1_des, pos2_des, vel1_des, vel2_des)).T

dt = 0.001
t_final = 11
x0 = [0.0, 0.0, 0.0, 0.0]

# controller parameters
Q = np.diag([30., 20., 8, 8])
R = 16*np.eye(1)
Qf = np.diag([11.67, 3.87, 0.1, 0.11])
Rf = 0.18*np.eye(1)

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = TVLQRController(csv_path=csv_path,
                             urdf_path=urdf_path,
                             dt=dt,
                             max_t = t_final,
                             torque_limit=torque_limit,
                             robot=robot)

controller.set_cost_parameters(Q=Q, R=R, Qf=Qf, Rf=Rf)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                T_des=T_des,
                X_des=X_des,
                U_des=U_des)
