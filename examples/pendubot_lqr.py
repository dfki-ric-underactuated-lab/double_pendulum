import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
#cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.02522]
torque_limit = [4.0, 0.0]

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

dt = 0.01
t_final = 2.0

controller = LQRController(mass=mass,
                           length=length,
                           com=com,
                           damping=damping,
                           gravity=gravity,
                           coulomb_fric=cfric,
                           inertia=inertia,
                           torque_limit=torque_limit)

controller.set_goal([np.pi, 0., 0., 0.])
# [0.01251931 6.51187283 6.87772744 9.35785251 0.02354949]
# [8.74006242e+01 1.12451099e-02 9.59966065e+01 8.99725246e-01 2.37517689e-01]
# [1.16402700e+01 7.95782007e+01 7.29021272e-02 3.02202319e-04 1.29619149e-01]
controller.set_cost_parameters(p1p1_cost=11.64,
                               p2p2_cost=79.58,
                               v1v1_cost=0.073,
                               v2v2_cost=0.0003,
                               p1v1_cost=0.,
                               p1v2_cost=0.,
                               p2v1_cost=0.,
                               p2v2_cost=0.,
                               u1u1_cost=0.13,
                               u2u2_cost=0.13,
                               u1u2_cost=0.)
controller.set_parameters(failure_value=0.0)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=[2.9, 0.25, 0.0, 0.0],  # 2.83, 0.35
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]])

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
