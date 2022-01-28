import os
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import SymbolicPFLAndLQRController


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [6.0, 0.0]


double_pendulum = SymbolicDoublePendulum(mass=mass,
                                         length=length,
                                         com=com,
                                         damping=damping,
                                         gravity=gravity,
                                         coulomb_fric=cfric,
                                         inertia=inertia,
                                         torque_limit=torque_limit)


controller = SymbolicPFLAndLQRController(mass,
                                         length,
                                         com,
                                         damping,
                                         gravity,
                                         cfric,
                                         inertia,
                                         torque_limit,
                                         "pendubot",
                                         "noncollocated")

sim = Simulator(plant=double_pendulum)

controller.set_goal([np.pi, 0, 0, 0])

# par = [6.97474837, 9.84031538, 9.1297417]
par = [10.0, 10.0, 10.0]
#par = np.loadtxt(os.path.join("data",
#                              sorted(os.listdir("data"))[-1],
#                              "controller_par.csv"))
print(par)

controller.set_cost_parameters(kpos=par[0],
                               kvel=par[1],
                               ken=par[2])

dt = 0.01
t_final = 10.0

print("dt: ", dt)
print("t final: ", t_final)

controller.init()

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

plot_timeseries(T, X, U, energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                energy_y_lines=[des_energy])

# Save Trajectory to a csv file to be sent to the motor.

#csv_data = np.vstack((T, np.asarray(X).T[0], np.asarray(X).T[1], U)).T
#np.savetxt("traj_opt_traj.csv", csv_data, delimiter=',', header="time,pos,vel,torque", comments="")
