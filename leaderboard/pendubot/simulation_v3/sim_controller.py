import os
import argparse
import importlib
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries

from sim_parameters import mpar, mpar_nolim, dt, t_final, t0, x0, integrator


def simulate_controller(controller, save_dir, controller_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plant = DoublePendulumPlant(model_pars=mpar_nolim)
    sim = Simulator(plant=plant)

    T, X, U = sim.simulate_and_animate(
        t0=t0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        save_video=True,
        video_name=os.path.join(save_dir, "sim_video.gif"),
        plot_horizontal_line=True,
        horizontal_line_height=0.9 * (mpar.l[0] + mpar.l[1]),
        scale=0.25,
    )

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
        show=False,
        scale=0.5,
    )

    if os.path.exists(f"readmes/{controller_name}.md"):
        os.system(f"cp readmes/{controller_name}.md {save_dir}/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", help="name of the controller to simulate")
    controller_arg = parser.parse_args().controller
    if controller_arg[-3:] == ".py":
        controller_arg = controller_arg[:-3]

    controller_name = controller_arg[4:]
    print(f"Simulating controller {controller_name}")

    if not os.path.exists("data"):
        os.makedirs("data")
    save_dir = f"data/{controller_name}"

    imp = importlib.import_module(controller_arg)

    controller = imp.controller

    simulate_controller(controller, save_dir)
