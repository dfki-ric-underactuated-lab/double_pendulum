import numpy as np
# import sys
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
# sys.path.append("../model")

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 10.0]

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
t_final = 10.0

T, X, U = sim.simulate_and_animate(t0=0.0, x0=[3.0, 0.1, 0.0, 0.0],
                                   tf=t_final, dt=dt, integrator="runge_kutta")

fig, ax = plt.subplots(3, 1, figsize=(18, 6), sharex="all")

ax[0].plot(T, np.asarray(X).T[0], label="q1")
ax[0].plot(T, np.asarray(X).T[1], label="q2")
ax[0].set_ylabel("angle [rad]")
ax[0].legend(loc="best")
ax[1].plot(T, np.asarray(X).T[2], label="q1 dot")
ax[1].plot(T, np.asarray(X).T[3], label="q2 dot")
ax[1].set_ylabel("angular velocity [rad/s]")
ax[1].legend(loc="best")
ax[2].plot(T, np.asarray(U).T[0], label="u1")
ax[2].plot(T, np.asarray(U).T[1], label="u2")
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("input torque [Nm]")
ax[2].legend(loc="best")
plt.show()