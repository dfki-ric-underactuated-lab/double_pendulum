import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries

# tmotors
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 4.0]

# mjbots
# mass = [0.608, 0.6268]
# length = [0.3, 0.2]
# com = [0.317763, 0.186981]
# damping = [7.480e-02, 1.602e-08]
# # cfric = [7.998e-02, 6.336e-02]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.0280354, 0.0183778]
# torque_limit = [0.0, 10.0]


plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

dt = 0.005
t_final = 10.0

controller = LQRController(mass=mass,
                           length=length,
                           com=com,
                           damping=damping,
                           gravity=gravity,
                           coulomb_fric=cfric,
                           inertia=inertia,
                           torque_limit=torque_limit)

controller.set_goal([np.pi, 0., 0., 0.])
# 11.67044417  0.10076462  3.86730188  0.11016735  0.18307841
controller.set_cost_parameters(p1p1_cost=11.67,
                               p2p2_cost=3.87,
                               v1v1_cost=0.10,
                               v2v2_cost=0.11,
                               p1v1_cost=0.,
                               p1v2_cost=0.,
                               p2v1_cost=0.,
                               p2v2_cost=0.,
                               u1u1_cost=0.18,
                               u2u2_cost=0.18,
                               u1u2_cost=0.)
controller.set_parameters(failure_value=0.0)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=[2.32, 1.87, 0.0, 0.0],
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   save_video=False,
                                   video_name="data/acrobot/lqr/acrobot_lqr")

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi])#,
                #tau_y_lines=[-torque_limit[1], torque_limit[1]])

# fig, ax = plt.subplots(3, 1, figsize=(18, 6), sharex="all")
# 
# ax[0].plot(T, np.asarray(X).T[0], label="q1")
# ax[0].plot(T, np.asarray(X).T[1], label="q2")
# ax[0].set_ylabel("angle [rad]")
# ax[0].legend(loc="best")
# ax[1].plot(T, np.asarray(X).T[2], label="q1 dot")
# ax[1].plot(T, np.asarray(X).T[3], label="q2 dot")
# ax[1].set_ylabel("angular velocity [rad/s]")
# ax[1].legend(loc="best")
# ax[2].plot(T, np.asarray(U).T[0], label="u1")
# ax[2].plot(T, np.asarray(U).T[1], label="u2")
# ax[2].set_xlabel("time [s]")
# ax[2].set_ylabel("input torque [Nm]")
# ax[2].legend(loc="best")
# plt.show()
