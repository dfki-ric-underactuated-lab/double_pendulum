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
from double_pendulum.simulation.perturbations import (
    get_gaussian_perturbation_array,
    get_random_gauss_perturbation_array,
    plot_perturbation_array,
)

SAVE = "save" in sys.argv
PLOT = "plot" in sys.argv
ANIMATE = "animate" in sys.argv

design = "design_C.0"
model = "model_3.0"
traj_model = "model_3.1"
robot = "acrobot"

friction_compensation = True

# model parameters
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
elif robot == "pendubot":
    torque_limit = [6.0, 0.0]
else:
    torque_limit = [6.0, 6.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
# mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit)

# swingup parameters
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# simulation parameter
dt = 0.005
t_final = 10.0  # 4.985
integrator = "runge_kutta"

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
delay_mode = "posvel"
delay = 0.0
u_noise_sigmas = [0.0, 0.0]
u_responsiveness = 1.0
perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
    t_final, dt, 3, 1.0, [0.01, 0.05], [0.1, 1.0]
)
# plot_perturbation_array(t_final, dt, perturbation_array)

# filter args
lowpass_alpha = [1.0, 1.0, 0.3, 0.3]
filter_velocity_cut = 0.1

# controller parameters
N = 100
con_dt = dt
N_init = 1000
max_iter = 10
max_iter_init = 1000
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = True
shifting = 1

# trajectory parameters
init_csv_path = os.path.join(
    "../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

if robot == "acrobot":
    sCu = [0.1, 0.1]
    sCp = [0.1, 0.1]
    sCv = [0.01, 0.1]
    sCen = 0.0
    fCp = [100.0, 10.0]
    fCv = [10.0, 1.0]
    fCen = 0.0

    f_sCu = [0.1, 0.1]
    f_sCp = [0.1, 0.1]
    f_sCv = [0.01, 0.01]
    f_sCen = 0.0
    f_fCp = [10.0, 10.0]
    f_fCv = [1.0, 1.0]
    f_fCen = 0.0

elif robot == "pendubot":
    sCu = [0.001, 0.001]
    sCp = [0.01, 0.01]
    sCv = [0.01, 0.01]
    sCen = 0.0
    fCp = [100.0, 100.0]
    fCv = [1.0, 1.0]
    fCen = 0.0

    f_sCu = sCu
    f_sCp = sCp
    f_sCv = sCv
    f_sCen = sCen
    f_fCp = fCp
    f_fCv = fCv
    f_fCen = fCen
else:
    sCu = [0.001, 0.001]
    sCp = [0.01, 0.01]
    sCv = [0.01, 0.01]
    sCen = 0.0
    fCp = [100.0, 100.0]
    fCv = [1.0, 1.0]
    fCen = 0.0

    f_sCu = sCu
    f_sCp = sCp
    f_sCv = sCv
    f_sCen = sCen
    f_fCp = fCp
    f_fCv = fCv
    f_fCen = fCen

init_sCu = sCu
init_sCp = sCp
init_sCv = sCv
init_sCen = sCen
init_fCp = fCp
init_fCv = fCv
init_fCen = fCen

# save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "ilqr_mpc", timestamp)
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
sim.set_disturbances(perturbation_array)

# filter
filter = lowpass_filter(lowpass_alpha, start, filter_velocity_cut)

# controller
controller = ILQRMPCCPPController(model_pars=mpar_con)
controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(
    N=N,
    dt=con_dt,
    max_iter=max_iter,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    integrator=integrator,
    trajectory_stabilization=trajectory_stabilization,
    shifting=shifting,
)
controller.set_cost_parameters(
    sCu=sCu, sCp=sCp, sCv=sCv, sCen=sCen, fCp=fCp, fCv=fCv, fCen=fCen
)
controller.set_final_cost_parameters(
    sCu=f_sCu, sCp=f_sCp, sCv=f_sCv, sCen=f_sCen, fCp=f_fCp, fCv=f_fCv, fCen=f_fCen
)
controller.set_filter(filter)
if init_csv_path is None:
    controller.compute_init_traj(
        N=N_init,
        dt=dt,
        max_iter=max_iter_init,
        regu_init=regu_init,
        max_regu=max_regu,
        min_regu=min_regu,
        break_cost_redu=break_cost_redu,
        sCu=init_sCu,
        sCp=init_sCp,
        sCv=init_sCv,
        sCen=init_sCen,
        fCp=init_fCp,
        fCv=init_fCv,
        fCen=init_fCen,
        integrator=integrator,
    )
else:
    controller.load_init_traj(csv_path=init_csv_path, num_break=40, poly_degree=3)

if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)

controller.init()

if ANIMATE:
    T, X, U = sim.simulate_and_animate(
        t0=0.0,
        x0=start,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        plot_inittraj=True,
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
        integrator=integrator,
    )

# saving and plotting
save_plot_to = None
if SAVE:
    os.system(f"cp {init_csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
    filter.save(save_dir)
    mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
    controller.save(save_dir)
    save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)
    save_plot_to = os.path.join(save_dir, "timeseries")

T_des, X_des, U_des = controller.get_init_trajectory()

if PLOT or SAVE:
    plot_timeseries(
        T,
        X,
        U,
        None,
        plot_energy=False,
        pos_y_lines=[0.0, np.pi],
        tau_y_lines=[-torque_limit[1], torque_limit[1]],
        T_des=T_des,
        X_des=X_des,
        U_des=U_des,
        save_to=save_plot_to,
        show=PLOT,
    )
