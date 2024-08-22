import sys
import os
from datetime import datetime
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)
from double_pendulum.filter.lowpass import lowpass_filter

PLOT = "plot" in sys.argv
ANIMATE = "animate" in sys.argv

## model parameters
robot = "pendubot"
# robot = "acrobot"

friction_compensation = True
stabilization = "lqr"

if robot == "pendubot":
    # design = "design_A.0"
    # model = "model_2.0"
    # load_path = "../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr"
    # model_path = "../../data/policies/design_A.0/model_2.0/pendubot/SAC/sac_model.zip"
    # scaling_state = True
    # torque_limit = [5.0, 0.5]
    # active_act = 0

    design = "design_C.1"
    model = "model_1.0"
    scaling_state = False
    torque_limit = [5.0, 0.0]
    active_act = 0
    load_path = "../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr/"
    model_path = "../../data/policies/design_C.1/model_1.0/pendubot/SAC/best_model.zip"  # about 40% success rate

elif robot == "acrobot":
    # design = "design_C.0"
    # model = "model_3.0"
    # load_path = "../../data/controller_parameters/design_C.0/acrobot/lqr/roa"
    # model_path = "../../data/policies/design_C.0/model_3.0/acrobot/SAC/sac_model.zip"
    # scaling_state = True
    # torque_limit = [0.5, 5.0]
    # active_act = 1

    design = "design_C.1"
    model = "model_1.0"
    torque_limit = [0.0, 5.0]
    active_act = 1
    scaling_state = True
    load_path = "../../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr/"
    model_path = "../../data/policies/design_C.1/model_1.0/acrobot/SAC/sac_model.zip"

# import model
model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
# model for simulation
mpar = model_parameters(filepath=model_par_path)
# model for controller
mpar_con = model_parameters(filepath=model_par_path)

if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.0025
t_final = 10.0
integrator = "runge_kutta"

# swingup parameters
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.5, 0.5]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0.0, 0.0]
# u_responsiveness = 1.0
u_responsiveness = 0.85
perturbation_times = []
perturbation_taus = []

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.1

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
# switching conditions
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))
flag = False

# LQR parameters
lqr_pars = np.loadtxt(os.path.join(load_path, "controller_par.csv"))
Q = np.diag(lqr_pars[:4])
R = np.diag([lqr_pars[4], lqr_pars[4]])


def condition1(t, x):
    return False


def check_if_state_in_roa(S, rho, x):
    # print(x)
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    # print(rad, rho)
    return rad < 1.0 * rho, rad


def condition2(t, x):
    # print("x=",x)
    y = wrap_angles_top(x)
    # y = wrap_angles_top()
    # print("y=",y)
    flag, rad = check_if_state_in_roa(S, rho, y)
    # print(rad, rho)
    if flag:
        print(t)
        print(y)
        print(flag)
        return flag
    return flag


# def condition2(t,x):
#     return False

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    scaling=False,
)

# initialize sac controller
controller1 = SACController(
    model_path=model_path, dynamics_func=dynamics_func, dt=dt, scaling=scaling_state
)

# initialize lqr controller
controller2 = LQRController(model_pars=mpar_con)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=15)

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=scaling_state,
)

if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
    # controller.set_friction_compensation(damping=[0., mpar.b[1]], coulomb_fric=[0., mpar.cf[1]])
controller.set_filter(filter)
controller.init()

if ANIMATE:
    T, X, U = sim.simulate_and_animate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        # save_video=True,
        # video_name="pendubot_designC.1_noisy.mp4"
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

if PLOT:
    plot_timeseries(
        T,
        X,
        U,
        None,
        plot_energy=False,
        X_filt=filter.x_filt_hist,
        X_meas=sim.meas_x_values,
        U_con=controller.u_hist,
        U_friccomp=controller.u_fric_hist,
        pos_y_lines=[0.0, np.pi],
        tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
        # save_to=os.path.join(save_dir, "timeseries"
        # save_to=os.path.join("pendubot_noisy.png")
    )

