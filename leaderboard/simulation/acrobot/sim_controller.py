import os
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries

from sim_parameters import mpar, dt, t_final, t0, x0, goal, integrator

parser = argparse.ArgumentParser()
parser.add_argument("controller", help="name of the controller to simulate")
controller_arg = parser.parse_args().controller
if controller_arg[-3:] == ".py":
    controller_arg = controller_arg[:-3]

controller_name = controller_arg[4:]
print(f"Simulating controller {controller_name}")

save_dir = f"data/{controller_name}"

imp = importlib.import_module(controller_arg)

controller = imp.controller

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

T, X, U = sim.simulate(t0=t0, x0=x0, tf=t_final, dt=dt, controller=controller, integrator=integrator)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_trajectory(os.path.join(save_dir, "sim_swingup.csv"), T=T, X_meas=X, U_con=U)

plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[-np.pi, 0.0, np.pi],
    vel_y_lines=[0.0],
    tau_y_lines=[-mpar.tl[1], 0.0, mpar.tl[1]],
    save_to=os.path.join(save_dir, "timeseries"),
)
