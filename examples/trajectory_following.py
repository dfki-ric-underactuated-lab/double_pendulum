import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController
from double_pendulum.utils.plotting import plot_timeseries


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
#cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 6.0]

csv_path = "data/ilqr/trajectory.csv"

dt = 0.005
t_final = 5.0

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=csv_path,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=[0.0, 0.0, 0.0, 0.0],
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi])#,
