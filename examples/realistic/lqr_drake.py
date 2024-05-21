import os
import sys
from datetime import datetime
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller_drake import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.filter.lowpass import lowpass_filter

SAVE = "save" in sys.argv
PLOT = "plot" in sys.argv
ANIMATE = "animate" in sys.argv

# model parameters
design = "design_A.0"
model = "model_2.0"
robot = "pendubot"

urdf_path = "../../data/urdfs/design_A.0/model_1.0/" + robot + ".urdf"
friction_compensation = True

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
else:
    torque_limit = [5.0, 5.0]
    active_act = 1

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

# simulation parameters
dt = 0.002
t_final = 4.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0.0, 0.0]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.0

if robot == "acrobot":
    x0 = [np.pi + 0.05, -0.2, 0.0, 0.0]
    Q = np.eye(4)
    R = np.eye(1)
elif robot == "pendubot":
    x0 = [2.9, 0.3, 0.0, 0.0]
    Q = np.diag([11.64, 79.58, 0.073, 0.0003])
    R = np.eye(1) * 0.13
else:
    x0 = [2.9, 0.3, 0.0, 0.0]
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.eye(1)

# save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "lqr_drake", timestamp)
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
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

# controller
controller = LQRController(
    urdf_path=urdf_path, model_pars=mpar_con, robot=robot, torque_limit=torque_limit
)
controller.set_goal([np.pi, 0.0, 0.0, 0.0])
controller.set_cost_matrices(Q=Q, R=R)
controller.set_filter(filter)

if friction_compensation:
    controller.set_friction_compensation(damping=[0.0, 0.0], coulomb_fric=mpar.cf)
controller.init()

if ANIMATE:
    T, X, U = sim.simulate_and_animate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        save_video=False,
        video_name=os.path.join(save_dir, "simulation"),
    )
else:
    T, X, U = sim.simulate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
    )

# saving and plotting
save_plot_to = None
if SAVE:
    os.system(f"mv {robot}.urdf " + os.path.join(save_dir, "{robot}.urdf"))
    filter.save(save_dir)
    mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
    controller.save(save_dir)
    save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)
    save_plot_to = os.path.join(save_dir, "timeseries")

if PLOT or SAVE:
    plot_timeseries(
        T,
        X,
        U,
        None,
        plot_energy=False,
        X_filt=filter.x_filt_hist,
        X_meas=sim.meas_x_values,
        U_con=controller.u_hist[1:],
        U_friccomp=controller.u_fric_hist,
        pos_y_lines=[0.0, np.pi],
        tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
        save_to=save_plot_to,
        show=PLOT,
    )
