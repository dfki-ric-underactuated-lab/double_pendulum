import os
import numpy as np
import pandas as pd

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController
from double_pendulum.controller.trajectory_following.pid_controller import PIDController
from double_pendulum.utils.plotting import plot_timeseries


robot = "acrobot"
trajopt = "ilqr"

mass = [0.608, 0.5]
length = [0.3, 0.4]
com = [length[0], length[1]]
damping = [0.081, 0.0]
#damping = [0., 0.]
cfric = [0.093, 0.186]
#cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
torque_limit = [12.0, 12.0]

# csv file
use_feed_forward_torque = True
read_with = "numpy"
latest_dir = sorted(os.listdir(os.path.join("data", robot, trajopt, "trajopt")))[-1]
csv_path = os.path.join("data", robot, trajopt, "trajopt", latest_dir, "trajectory.csv")
if read_with == "numpy":
    trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")

    x0 = trajectory[0][1:5]

    time_traj = trajectory.T[0]
    p1_traj = trajectory.T[1]
    p2_traj = trajectory.T[2]
    v1_traj = trajectory.T[3]
    v2_traj = trajectory.T[4]
    if use_feed_forward_torque:
        u1_traj = trajectory.T[5]
        u2_traj = trajectory.T[6]
    else:
        u1_traj = np.zeros_like(time_traj)
        u2_traj = np.zeros_like(time_traj)

elif read_with == "pandas":
    trajectory = pd.read_csv(csv_path)

    time_traj = np.asarray(trajectory["time"])
    p1_traj = np.asarray(trajectory["shoulder_pos"])
    p2_traj = np.asarray(trajectory["elbow_pos"])
    v1_traj = np.asarray(trajectory["shoulder_vel"])
    v2_traj = np.asarray(trajectory["elbow_vel"])
    if use_feed_forward_torque:
        tau1_traj = np.asarray(trajectory["shoulder_torque"])
        tau2_traj = np.asarray(trajectory["elbow_torque"])
    else:
        u1_traj = np.zeros_like(time_traj)
        u2_traj = np.zeros_like(time_traj)

dt = time_traj[1] - time_traj[0]
t_final = time_traj[-1]
x0 = np.asarray([p1_traj[0],
                 p2_traj[0],
                 v1_traj[0],
                 v2_traj[0]])

T_des = time_traj
X_des = np.asarray([p1_traj, p2_traj, v1_traj, v2_traj]).T
U_des = np.asarray([u1_traj, u2_traj]).T
# dt = 0.005
# t_final = 5.0

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

# controller = TrajectoryController(csv_path=csv_path,
#                                   torque_limit=torque_limit,
#                                   kK_stabilization=True)
controller = PIDController(csv_path=csv_path,
                           read_with=read_with,
                           use_feed_forward_torque=use_feed_forward_torque,
                           torque_limit=torque_limit)
controller.set_parameters(Kp=200.0, Ki=0.0, Kd=2.0)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   plot_forecast=False)
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                T_des=T_des, X_des=X_des, U_des=U_des)
