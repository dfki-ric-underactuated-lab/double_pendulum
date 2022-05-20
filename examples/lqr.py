import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries


robot = "acrobot"

# tmotors
# mass = [0.608, 0.630]
# length = [0.3, 0.2]
# com = [0.275, 0.166]
# damping = [0.081, 0.0]
# # cfric = [0.093, 0.186]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.05472, 0.02522]
# torque_limit = [0.0, 4.0]

# mjbots
# mass = [0.608, 0.6268]
# length = [0.3, 0.2]
# com = [0.317763, 0.186981]
# damping = [7.480e-02, 1.602e-08]
# # cfric = [7.998e-02, 6.336e-02]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.0280354, 0.0183778]
# torque_limit = [0.0, 10.0]

# mass = [0.608, 0.6]
# length = [0.3, 0.4]
# com = [length[0], length[1]]
# #damping = [0.0, 0.0]
# #damping = [0.0, 0.01]
# damping = [0.0005, 0.0001]
# cfric = [0.093, 0.186]
# #cfric = [0.0, 0.186]
# #cfric = [0., 0.]
# gravity = 9.81
# inertia = [mass[0]*length[0]**2., mass[1]*length[1]**2.]

cfric = [0., 0.]
motor_inertia = 0.
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
# model_par_path = "../data/system_identification/identified_parameters/tmotors_v2.0/model_parameters_est.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
# mpar.set_damping(damping)
mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 4.0
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]

# imperfections = False
#noise_mode = "vel"
#noise_amplitude = 0.0
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_cut = 0.0
meas_noise_vfilter = "none"
meas_noise_vfilter_args = {"alpha": [1., 1., 0.3, 0.3],
                           "kalman":{"x_lin": goal, "u_lin": [0., 0.]}}
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0., 0.]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []


if robot == "acrobot":
    # x0 = [2.85, 0.7, 0.0, 0.0]
    # x0 = [-2.85, -0.7, 0.0, 0.0]
    # x0 = [-2.95, -0.4, 0.0, 0.0]
    # x0 = [2.95, 0.4, 0.0, 0.0]
    # x0 = [2.95, 0.0, 0.0, 0.0]
    # x0 = [-2.95, 0.0, 0.0, 0.0]
    # x0 = [-3.1, -0.3, 0.0, 0.0]
    # x0 = [3.1, 0.3, 0.0, 0.0]
    # x0 = [3.1, 0.4, 0.0, 0.0]
    # x0 = [-3.1, -0.4, 0.0, 0.0]
    x0 = [np.pi+0.05, -0.2, 0.0, 0.0]

    # 11.67044417  0.10076462  3.86730188  0.11016735  0.18307841
    # [6.60845661 0.0203263  1.59384209 0.11347305 0.08903846]
    # 1.649609700742603735e+01 9.094310297259731612e+01 7.128663050519863653e-02 1.116726623434960083e-02 3.647472178659907360e+00
    #c[7.568227051118126880e+00, 1.851805841833500610e+00, 9.989157089721836247e-01, 9.994476149737525628e-01, 7.567700329462909714e-01]
    #Q = np.diag((0.97, 0.93, 0.39, 0.26))
    #R = np.diag((0.11, 0.11))
    #Q = np.diag((0.5, 0.5, 0.01, 0.01))
    #R = np.diag((0.11, 0.11))

    # tvlqr costs
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = np.eye(2)*0.82

elif robot == "pendubot":
    x0 = [2.9, 0.3, 0.0, 0.0]

    # [0.01251931 6.51187283 6.87772744 9.35785251 0.02354949]
    # [8.74006242e+01 1.12451099e-02 9.59966065e+01 8.99725246e-01 2.37517689e-01]
    # [1.16402700e+01 7.95782007e+01 7.29021272e-02 3.02202319e-04 1.29619149e-01]
    Q = np.diag((11.64, 79.58, 0.073, 0.0003))
    R = np.diag((0.13, 0.13))

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "lqr", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                               delay=delay,
                               delay_mode=delay_mode)
sim.set_filter_parameters(meas_noise_cut=meas_noise_cut,
                          meas_noise_vfilter=meas_noise_vfilter,
                          meas_noise_vfilter_args=meas_noise_vfilter_args)
sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                         u_responsiveness=u_responsiveness)

controller = LQRController(model_pars=mpar)
controller.set_goal(goal)
controller.set_cost_matrices(Q=Q, R=R)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1e9)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"))
X_meas = sim.meas_x_values
X_filt = sim.filt_x_values
U_con = sim.con_u_values

plot_timeseries(T, X, U, None,
                plot_energy=False,
                #T_des=T[1:],
                #X_des=X_filt,
                X_meas=X_meas,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "time_series"))
