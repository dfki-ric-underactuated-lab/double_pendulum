import os
import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.pfl import EnergyShapingPFLAndLQRController

mass = [1.0, 1.0]
length = [1.0, 1.0]
damping = [0.0, 0.0]
gravity = 9.8
com = [0.5, 0.5]
coulomb_fric = [0.0, 0.0]
inertia = [0.2, 1.0]
torque_limit = [0.0, 20.0]

double_pendulum = SymbolicDoublePendulum(mass=mass,
                                         length=length,
                                         com=com,
                                         damping=damping,
                                         gravity=gravity,
                                         coulomb_fric=coulomb_fric,
                                         inertia=inertia,
                                         torque_limit=torque_limit)


controller = EnergyShapingPFLAndLQRController(mass,
                                              length,
                                              com,
                                              damping,
                                              gravity,
                                              coulomb_fric,
                                              inertia,
                                              torque_limit)

sim = Simulator(plant=double_pendulum)

controller.set_goal([np.pi, 0, 0, 0])


par = [9.0, 3.0, 1.0]  # undamped
print(par)

controller.set_hyperpar(kpos=par[0],
                        kvel=par[1],
                        ken=par[2])

dt = 0.01
t_final = 6.0

print("dt: ", dt)
print("t final: ", t_final)

T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=[0.1, 0.0, 0.0, 0.0],
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator="runge_kutta",
                                   phase_plot=False,
                                   save_video=False)

#controller.save(path)
energy = controller.en
des_energy = controller.desired_energy

fig, ax = plt.subplots(4, 1, figsize=(18, 6), sharex="all")

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
ax[3].plot(T, np.asarray(energy), label="energy")
ax[3].plot([T[0], T[-1]], [des_energy, des_energy], label="energy", color="red")
ax[3].set_ylabel("energy [J]")
ax[3].legend(loc="best")
plt.show()

# Save Trajectory to a csv file to be sent to the motor.

#csv_data = np.vstack((T, np.asarray(X).T[0], np.asarray(X).T[1], U)).T
#np.savetxt("traj_opt_traj.csv", csv_data, delimiter=',', header="time,pos,vel,torque", comments="")
