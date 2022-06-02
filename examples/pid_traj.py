import numpy as np
import pandas as pd

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.utils.plotting import plot_timeseries

# model parameters
mass = [0.608, 0.22]
length = [0.2, 0.4]
com = [length[0], length[1]]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
gravity = 9.81
inertia = [mass[0]*length[0]**2., mass[1]*length[1]**2.]
torque_limit = [8.0, 8.0]

# trajectory
excitation_traj_csv = "../data/system_identification/excitation_trajectories/trajectory-pos-50.csv"
read_with = "pandas"

#simulation parameters
data = pd.read_csv(excitation_traj_csv)
time_traj = np.asarray(data["time"])
pos1_traj = np.asarray(data["shoulder_pos"])
pos2_traj = np.asarray(data["elbow_pos"])
vel1_traj = np.asarray(data["shoulder_vel"])
vel2_traj = np.asarray(data["elbow_vel"])

dt = time_traj[1] - time_traj[0]
t_final = time_traj[-1]
x0 = [pos1_traj[0], pos2_traj[0],
      vel1_traj[0], vel2_traj[0]]
integrator = "runge_kutta"

# controller parameters
Kp = 10.
Ki = 0.
Kd = 0.1

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = TrajPIDController(csv_path=excitation_traj_csv,
                               read_with=read_with,
                               use_feed_forward_torque=False,
                               torque_limit=torque_limit)

controller.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller.init()

# T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
#                                    tf=t_final, dt=dt, controller=controller,
#                                    integrator=integrator, phase_plot=False,
#                                    save_video=False)

T, X, U = sim.simulate(t0=0.0, x0=x0,
                       tf=t_final, dt=dt, controller=controller,
                       integrator=integrator)
T_des = time_traj.T
X_des = np.asarray([pos1_traj, pos2_traj,
                    vel1_traj, vel2_traj]).T

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[-np.pi, 0.0, np.pi],
                T_des=T_des,
                X_des=X_des)
