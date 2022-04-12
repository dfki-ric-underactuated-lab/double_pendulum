import os
from datetime import datetime
import numpy as np
import pandas as pd

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.saving import save_trajectory

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
csv_path = "data/acrobot/dircol/acrobot_tmotors_swingup_1000Hz.csv"
read_with = "pandas"

#csv_path = "data/acrobot/ilqr/trajopt/20220412-170342/trajectory.csv"
#read_with = "numpy"

# simulation parameters
x0 = [0.0, 0.0, 0.0, 0.0]

# controller parameters
Q = np.diag([1.0, 1.0, 0.1, 0.1])
R = 10.0*np.eye(1)
Qf = np.zeros((4,4))
# Qf = np.array([[6500., 1600., 1500.,  0.],
#                [1600.,  400.,  370.,  0.],
#                [1500.,  370.,  350.,  0.],
#                [   0.,    0.,    0., 30.]])

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

if read_with == "pandas":
    data = pd.read_csv(csv_path)
    time_traj = np.asarray(data["time"])
else:
    data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
    time_traj = data[:, 0]

dt = time_traj[1] - time_traj[0]
t_final = time_traj[-1]

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   plot_inittraj=True)

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "tvlqr", timestamp)
os.makedirs(save_dir)

os.system(f"cp {csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

if read_with == "pandas":
    data = pd.read_csv(csv_path)

    T_des = np.asarray(data["time"])
    pos1_des = np.asarray(data["shoulder_pos"])
    pos2_des = np.asarray(data["elbow_pos"])
    vel1_des = np.asarray(data["shoulder_vel"])
    vel2_des = np.asarray(data["elbow_vel"])
    tau1_des = np.asarray(data["shoulder_torque"])
    tau2_des = np.asarray(data["elbow_torque"])

elif read_with == "numpy":
    data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

    T_des = data[:,0]
    pos1_des = data[:,1]
    pos2_des = data[:,2]
    vel1_des = data[:,3]
    vel2_des = data[:,4]
    tau1_des = data[:,5]
    tau2_des = data[:,6]


U_des = np.vstack((tau1_des, tau2_des)).T
X_des = np.vstack((pos1_des, pos2_des, vel1_des, vel2_des)).T

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                T_des=T_des,
                X_des=X_des,
                U_des=U_des)

