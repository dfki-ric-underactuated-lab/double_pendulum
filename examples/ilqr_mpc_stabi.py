import os
from datetime import datetime
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory

robot = "pendubot"
friction_compensation = True

if robot == "acrobot":
    torque_limit = [0.0, 6.0]
    active_act = 0
if robot == "pendubot":
    torque_limit = [6.0, 0.0]
    active_act = 1

model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
# model_par_path = "../data/system_identification/identified_parameters/tmotors_v2.0/model_parameters_est.yml"
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
#mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0., 0.])
    mpar_con.set_cfric([0., 0.])
mpar_con.set_torque_limit(torque_limit)

# simulation parameter
dt = 0.005
t_final = 10.0  # 4.985
integrator = "runge_kutta"

process_noise_sigmas = [0., 0., 0., 0.]
meas_noise_sigmas = [0., 0., 0.05, 0.05]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0., 0.]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
meas_noise_cut = 0.0
meas_noise_vfilter = "none"
filter_kwargs = {"lowpass_alpha": [1., 1., 0.3, 0.3],
                 "kalman_xlin": [np.pi, 0., 0., 0.],
                 "kalman_ulin": [0., 0.],
                 "kalman_process_noise_sigmas": process_noise_sigmas,
                 "kalman_meas_noise_sigmas": meas_noise_sigmas,
                 "ukalman_integrator": integrator,
                 "ukalman_process_noise_sigmas": process_noise_sigmas,
                 "ukalman_meas_noise_sigmas": meas_noise_sigmas}

# controller parameters
# N = 20
N = 100
con_dt = dt
N_init = 100
max_iter = 20
max_iter_init = 1000
regu_init = 1.
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = False
shifting = 1

# swingup parameters
start = [np.pi+0.05, -0.2, 0., 0.]
goal = [np.pi, 0., 0., 0.]


if robot == "acrobot":
    # sCu = [0.1, 0.1]
    # sCp = [.1, .01]
    # sCv = [.1, .01]
    # sCen = 0.0
    # fCp = [10., 1.]
    # fCv = [10., 1.]
    # fCen = 0.0

    sCu = [0.1, 0.1]
    sCp = [.1, .1]
    sCv = [.01, .01]
    sCen = 0.0
    fCp = [10., 10.]
    fCv = [1., 1.]
    fCen = 0.0

if robot == "pendubot":
    sCu = [0.0001, 0.0001]
    sCp = [0.1, 0.1]
    sCv = [0.01, 0.01]
    sCen = 0.
    fCp = [10., 10.]
    fCv = [.1, .1]
    fCen = 0.

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "ilqr", "mpc", timestamp)
os.makedirs(save_dir)

# construct simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                               delay=delay,
                               delay_mode=delay_mode)
sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                         u_responsiveness=u_responsiveness)

controller = ILQRMPCCPPController(model_pars=mpar_con)
# controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(N=N,
                          dt=con_dt,
                          max_iter=max_iter,
                          regu_init=regu_init,
                          max_regu=max_regu,
                          min_regu=min_regu,
                          break_cost_redu=break_cost_redu,
                          integrator=integrator,
                          trajectory_stabilization=trajectory_stabilization,
                          shifting=shifting)
controller.set_cost_parameters(sCu=sCu,
                               sCp=sCp,
                               sCv=sCv,
                               sCen=sCen,
                               fCp=fCp,
                               fCv=fCv,
                               fCen=fCen)
# controller.compute_init_traj(N=N_init,
#                              dt=dt,
#                              max_iter=max_iter_init,
#                              regu_init=regu_init,
#                              max_regu=max_regu,
#                              min_regu=min_regu,
#                              break_cost_redu=break_cost_redu,
#                              sCu=init_sCu,
#                              sCp=init_sCp,
#                              sCv=init_sCv,
#                              sCen=init_sCen,
#                              fCp=init_fCp,
#                              fCv=init_fCv,
#                              fCen=init_fCen,
#                              integrator=integrator)

controller.set_filter_args(filt=meas_noise_vfilter, x0=goal, dt=dt, plant=plant,
                           simulator=sim, velocity_cut=meas_noise_cut,
                           filter_kwargs=filter_kwargs)
if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)

controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=True, plot_forecast=True,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   anim_dt=5*dt)

# T, X, U = sim.simulate(t0=0.0, x0=start,
#                        tf=t_final, dt=dt, controller=controller,
#                        integrator="runge_kutta", imperfections=imperfections)

# saving and plotting
mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

par_dict = {
            "dt": dt,
            "t_final": t_final,
            "integrator": integrator,
            "start_pos1": start[0],
            "start_pos2": start[1],
            "start_vel1": start[2],
            "start_vel2": start[3],
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3],
            "N": N,
            "N_init": N_init,
            "max_iter": max_iter,
            "max_iter_init": max_iter_init,
            "regu_init": regu_init,
            "max_regu": max_regu,
            "min_regu": min_regu,
            "break_cost_redu": break_cost_redu,
            "trajectory_stabilization": trajectory_stabilization,
            "sCu1": sCu[0],
            "sCu2": sCu[1],
            "sCp1": sCp[0],
            "sCp2": sCp[1],
            "sCv1": sCv[0],
            "sCv2": sCv[1],
            "sCen": sCen,
            "fCp1": fCp[0],
            "fCp2": fCp[1],
            "fCv1": fCv[0],
            "fCv2": fCv[1],
            "fCen": fCen
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)

save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                X_filt=controller.x_filt_hist,
                X_meas=sim.meas_x_values,
                U_con=controller.u_hist,
                U_friccomp=controller.u_fric_hist,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "timeseries"))
