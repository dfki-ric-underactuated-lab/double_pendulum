import os
import sys
from datetime import datetime
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.perturbations import (
    get_gaussian_perturbation_array,
    get_random_gauss_perturbation_array,
    plot_perturbation_array,
)
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.filter.lowpass import lowpass_filter

SAVE = "save" in sys.argv
PLOT = "plot" in sys.argv
ANIMATE = "animate" in sys.argv

# model parameters
design = "design_C.0"
model = "model_3.0"
robot = "acrobot"
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
mpar_con.set_motor_inertia(0.0)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"

# swingup parameters
goal = [np.pi, 0.0, 0.0, 0.0]

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
delay_mode = "posvel"
delay = 0.0
u_noise_sigmas = [0.0, 0.0]
u_responsiveness = 1.0
# mu = [[1.0, 4.0, 7.0], [2.0, 6.0, 8.0]]
# sigma = [[0.05, 0.01, 0.05], [0.05, 0.01, 0.05]]
# amplitude = [[0.25, -0.5, 1.0], [0.25, -0.5, 1.0]]
# perturbation_array = get_gaussian_perturbation_array(t_final, dt, mu, sigma, amplitude)
perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
    t_final, dt, 3, 1.0, [0.01, 0.05], [0.1, 1.0]
)
# plot_perturbation_array(t_final, dt, perturbation_array)

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.0

if robot == "acrobot":
    x0 = [np.pi + 0.1, -0.4, 0.0, 0.0]
    Q = np.diag([0.64, 0.99, 0.78, 0.64])
    R = np.eye(2) * 0.27

elif robot == "pendubot":
    x0 = [np.pi - 0.2, 0.3, 0.0, 0.0]
    Q = np.diag([0.0125, 6.5, 6.88, 9.36])
    R = np.diag([2.4, 2.4])
else:
    x0 = [np.pi - 0.2, 0.3, 0.0, 0.0]
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.diag([1.0, 1.0])

# save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "lqr", timestamp)
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
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

# controller
controller = LQRController(model_pars=mpar_con)
controller.set_goal(goal)
controller.set_cost_matrices(Q=Q, R=R)
controller.set_parameters(failure_value=0.0, cost_to_go_cut=1000)
controller.set_filter(filter)

if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
controller.init()
# print(controller.S)
# print(controller.K)

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
