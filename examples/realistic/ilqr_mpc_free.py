import os
import sys
import numpy as np
from datetime import datetime

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.filter.lowpass import lowpass_filter

SAVE = "save" in sys.argv
PLOT = "plot" in sys.argv
ANIMATE = "animate" in sys.argv

design = "design_A.0"
model = "model_2.0"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 4.0]
elif robot == "pendubot":
    torque_limit = [4.0, 0.0]
else:
    torque_limit = [4.0, 4.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# swingup parameters
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# simulation parameter
dt = 0.005
t_final = 10.0
integrator = "runge_kutta"

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
delay_mode = "posvel"
delay = 0.01
u_noise_sigmas = [0.01, 0.01]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
lowpass_alpha = [1.0, 1.0, 0.3, 0.3]
filter_velocity_cut = 0.1

# controller parameters
N = 200
N_init = 1000
max_iter = 2
max_iter_init = 100
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.0001
break_cost_redu = 1e-6
trajectory_stabilization = False

if robot == "acrobot":
    f_sCu = [0.0001, 0.0001]
    f_sCp = [0.1, 0.1]
    f_sCv = [0.01, 0.5]
    f_sCen = 0.0
    f_fCp = [10.0, 10.0]
    f_fCv = [1.0, 1.0]
    f_fCen = 0.0
elif robot == "pendubot":
    f_sCu = [0.0001, 0.0001]
    f_sCp = [0.1, 0.2]
    f_sCv = [0.0, 0.0]
    f_sCen = 0.0
    f_fCp = [10.0, 10.0]
    f_fCv = [1.0, 1.0]
    f_fCen = 1.0
else:
    f_sCu = [0.0001, 0.0001]
    f_sCp = [0.1, 0.1]
    f_sCv = [0.0, 0.0]
    f_sCen = 0.0
    f_fCp = [10.0, 10.0]
    f_fCv = [1.0, 1.0]
    f_fCen = 0.0

# save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "ilqr_mpc_free", timestamp)
if SAVE:
    os.makedirs(save_dir)

# plant
plant = DoublePendulumPlant(model_pars=mpar)

# simulator
sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(
    meas_noise_sigmas=meas_noise_sigmas, delay=delay, delay_mode=delay_mode
)
sim.set_motor_parameters(
    u_noise_sigmas=u_noise_sigmas, u_responsiveness=u_responsiveness
)

# filter
filter = lowpass_filter(lowpass_alpha, start, filter_velocity_cut)

# controller
controller = ILQRMPCCPPController(model_pars=mpar)
controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(
    N=N,
    dt=dt,
    max_iter=max_iter,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    integrator=integrator,
    trajectory_stabilization=trajectory_stabilization,
)
controller.set_cost_parameters(
    sCu=f_sCu, sCp=f_sCp, sCv=f_sCv, sCen=f_sCen, fCp=f_fCp, fCv=f_fCv, fCen=f_fCen
)
controller.set_filter(filter)
controller.compute_init_traj(
    N=N_init,
    dt=dt,
    max_iter=max_iter_init,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    sCu=f_sCu,
    sCp=f_sCp,
    sCv=f_sCv,
    sCen=f_sCen,
    fCp=f_fCp,
    fCv=f_fCv,
    fCen=f_fCen,
    integrator=integrator,
)
controller.init()

if ANIMATE:
    T, X, U = sim.simulate_and_animate(
        t0=0.0,
        x0=start,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator="runge_kutta",
        plot_inittraj=False,
        plot_forecast=True,
        save_video=False,
        video_name=os.path.join(save_dir, "simulation"),
        anim_dt=0.02,
    )
else:
    T, X, U = sim.simulate(
        t0=0.0,
        x0=start,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator="runge_kutta",
    )


# saving and plotting
save_plot_to = None
if SAVE:
    mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
    filter.save(save_dir)
    controller.save(save_dir)
    save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)
    save_plot_to = os.path.join(save_dir, "timeseries")

if PLOT or SAVE:
    X_meas = sim.meas_x_values

    plot_timeseries(
        T,
        X,
        U,
        None,
        plot_energy=False,
        X_meas=X_meas,
        # X_filt=filt.x_filt_hist,
        pos_y_lines=[0.0, np.pi],
        tau_y_lines=[-torque_limit[1], torque_limit[1]],
        save_to=save_plot_to,
        show=PLOT,
    )
