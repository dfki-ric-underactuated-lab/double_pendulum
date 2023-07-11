import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)

## model parameters
# robot = "pendubot"
robot = "acrobot"

friction_compensation = True
stabilization = "lqr"

if robot == "pendubot":
    design = "design_A.0"
    model = "model_2.0"
    torque_limit = [5.0, 0.0]
    active_act = 0
    Q = 3.0 * np.diag([0.64, 0.64, 0.1, 0.1])
    R = np.eye(2) * 0.82
    load_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/lqr_data/pendubot/lqr/roa"
    model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/best_model/pendubot_model.zip"

elif robot == "acrobot":
    design = "design_C.0"
    model = "model_3.0"
    torque_limit = [0.0, 5.0]
    active_act = 1
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
    load_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/lqr_data/acrobot/lqr/roa"
    model_path = "/home/chi/Github/double_pendulum/examples/reinforcement_learning/SAC/best_model/acrobot_model.zip"

# import model
model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar_con = model_parameters(filepath=model_par_path)

if friction_compensation:
    mpar_con.set_damping([0., 0.])
    mpar_con.set_cfric([0., 0.])
mpar_con.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

## noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.5, 0.5]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0., 0.]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

## filter args
meas_noise_vfilter = "lowpass"
meas_noise_cut = 0.1
filter_kwargs = {"lowpass_alpha": [1., 1., 0.2, 0.2],
                 "kalman_xlin": goal,
                 "kalman_ulin": [0., 0.],
                 "kalman_process_noise_sigmas": process_noise_sigmas,
                 "kalman_meas_noise_sigmas": meas_noise_sigmas,
                 "ukalman_integrator": integrator,
                 "ukalman_process_noise_sigmas": process_noise_sigmas,
                 "ukalman_meas_noise_sigmas": meas_noise_sigmas}

sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                               delay=delay,
                               delay_mode=delay_mode)
sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                         u_responsiveness=u_responsiveness)
# switching conditions
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))
flag = False

def condition1(t, x):
    return False

def check_if_state_in_roa(S, rho, x):
    # print(x)
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    print(rad, rho)
    return rad < 1.0*rho, rad

def condition2(t, x):
    # print("x=",x)
    y = wrap_angles_top(x)
    # y = wrap_angles_top()
    # print("y=",y)
    flag,rad = check_if_state_in_roa(S,rho,y)
    print(rad,rho)
    if flag:
        print(t)
        print(y)
        print(flag)
        return flag
    return flag

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
)

# initialize sac controller
controller1 = SACController(
    model_path = model_path,
    dynamics_func=dynamics_func,
    dt=dt,
)

# initialize lqr controller
controller2 = LQRController(model_pars=mpar_con)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15)

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)

controller.set_filter_args(filt=meas_noise_vfilter, x0=goal, dt=dt, plant=plant,
                           simulator=sim, velocity_cut=meas_noise_cut,
                           filter_kwargs=filter_kwargs)
if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
    #controller.set_friction_compensation(damping=[0., mpar.b[1]], coulomb_fric=[0., mpar.cf[1]])
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

# plot timeseries
# plot_timeseries(
#     T,
#     X,
#     U,
#     X_meas=sim.meas_x_values,
#     pos_y_lines=[np.pi],
#     tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
#     # save_to="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/plots/acrobot.png"
# )

plot_timeseries(T, X, U, None,
                plot_energy=False,
                X_filt=controller.x_filt_hist,
                X_meas=sim.meas_x_values,
                U_con=controller.u_hist,
                U_friccomp=controller.u_fric_hist,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                # save_to=os.path.join(save_dir, "timeseries"
)