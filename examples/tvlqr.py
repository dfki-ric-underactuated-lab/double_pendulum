import os
from datetime import datetime
import numpy as np
import pandas as pd

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

# model parameters
urdf_path = "../data/urdfs/acrobot.urdf"
robot = "acrobot"

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.02522]
torque_limit = [0.0, 6.0]

# trajectory parameters
csv_path = "../data/trajectories/acrobot/dircol/acrobot_tmotors_swingup_1000Hz.csv"
read_with = "pandas  # for dircol traj"
keys = "shoulder-elbow"

# csv_path = "../data/trajectories/acrobot/ilqr/trajectory.csv"
# read_with = "numpy"
# keys = ""

# simulation parameters
x0 = [0.0, 0.0, 0.0, 0.0]

# controller parameters
Q = np.diag([10.0, 10.0, 1.0, 1.0])  # for dircol traj
R = 0.1*np.eye(1)
# Q = np.diag([100.0, 100.0, 10.0, 10.0]) # for ilqr traj
# R = 1.0*np.eye(1)

# Qf = np.zeros((4, 4))
Qf = np.copy(Q)
# Qf = np.array([[6500., 1600., 1500.,  0.],
#                [1600.,  400.,  370.,  0.],
#                [1500.,  370.,  350.,  0.],
#                [   0.,    0.,    0., 30.]])


# init plant, simulator and controller
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
                             read_with=read_with,
                             torque_limit=torque_limit,
                             robot=robot)

controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)
controller.init()

# load reference trajectory
T_des, X_des, U_des = load_trajectory(csv_path, read_with)
dt = T_des[1] - T_des[0]
t_final = T_des[-1]

# simulate
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   plot_inittraj=True)

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "tvlqr", timestamp)
os.makedirs(save_dir)

os.system(f"cp {csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                T_des=T_des,
                X_des=X_des,
                U_des=U_des)
