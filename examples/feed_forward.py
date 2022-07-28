import os
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.feed_forward import FeedForwardController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory


# model parameters
#cfric = [0., 0.]
#motor_inertia = 0.
torque_limit = [4.0, 4.0]
model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
#mpar.set_motor_inertia(motor_inertia)
#mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"
x0 = np.array([0., 0., 0., 0.])

# trajectory
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N+1)
u1 = 0.4*np.sin(10.*T_des)
# u1 = np.zeros(N+1)
u2 = 0.8*np.cos(10.*T_des)
U_des = np.array([u1, u2]).T

# simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller = FeedForwardController(T=T_des,
                                   U=U_des,
                                   torque_limit=torque_limit,
                                   num_break=40)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator, plot_inittraj=True,
                                   save_video=False)

# traj_file = os.path.join(save_dir, "trajectory.csv")
# save_trajectory(csv_path=traj_file,
#                 T=T, X=X, U=U)

plot_timeseries(T, X, U, None,
                #pos_y_lines=[-np.pi, 0.0, np.pi],
                T_des=T_des,
                U_des=U_des)
