import os
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries




def simulate_controller(controller, save_dir, mpar, dt, t_final, t0, x0, goal, integrator, controller_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plant = SymbolicDoublePendulum(model_pars=mpar)
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

