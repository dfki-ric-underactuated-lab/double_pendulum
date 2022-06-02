import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.utils.plotting import plot_timeseries

# model parameters
robot = "double_pendulum"
torque_limit = [5.0, 5.0]
model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
# model_par_path = "../data/system_identification/identified_parameters/tmotors_v2.0/model_parameters_est.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 4.0
integrator = "runge_kutta"
x0 = [np.pi-0.1, 0.2, 0., 0.2]
goal = [np.pi, 0., 0., 0.]

process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.5, 0.5]
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

# controller parameters
Kp = 10.
Ki = 0.
Kd = 0.1

# setup savedir
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "point_pid", timestamp)
os.makedirs(save_dir)

# setup simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(
        meas_noise_sigmas=meas_noise_sigmas,
        delay=delay,
        delay_mode=delay_mode)
sim.set_filter_parameters(
        meas_noise_cut=meas_noise_cut,
        meas_noise_vfilter=meas_noise_vfilter,
        meas_noise_vfilter_args=meas_noise_vfilter_args)
sim.set_motor_parameters(
        u_noise_sigmas=u_noise_sigmas,
        u_responsiveness=u_responsiveness)

controller = PointPIDController(
        torque_limit=torque_limit,
        goal=goal,
        dt=dt)
controller.set_parameters(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd)
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
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                save_to=os.path.join(save_dir, "time_series"))
