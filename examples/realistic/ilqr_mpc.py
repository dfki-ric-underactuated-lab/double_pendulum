import os
from datetime import datetime
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

robot = "acrobot"
friction_compensation = True

# # model parameters
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]

model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters_new2.yml"
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
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0., 0.]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
meas_noise_vfilter = "none"
meas_noise_cut = 0.
filter_kwargs = {"lowpass_alpha": [1., 1., 0.3, 0.3],
                 "kalman_xlin": goal,
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
N_init = 1000
max_iter = 10
max_iter_init = 1000
regu_init = 1.
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = True
shifting = 1

# trajectory parameters
## tmotors v1.0
# init_csv_path = "../data/trajectories/acrobot/dircol/acrobot_tmotors_swingup_1000Hz.csv"

## tmotors v1.0
# init_csv_path = "../data/trajectories/acrobot/ilqr_v1.0/trajectory2.csv"

# tmotors v2.0
# init_csv_path = "../data/trajectories/acrobot/ilqr/trajectory.csv"

#latest_dir = sorted(os.listdir(os.path.join("data", robot, "ilqr", "trajopt")))[-1]
#init_csv_path = os.path.join("data", robot, "ilqr", "trajopt", latest_dir, "trajectory.csv")
init_csv_path = os.path.join("../data/trajectories", robot, "ilqr_v1.0_new2/trajectory.csv")
#init_csv_path = "../data/trajectories/acrobot/ilqr/trajectory.csv"

if robot == "acrobot":
    # u_prefac = 1.
    # stage_prefac = 1.
    # final_prefac = 1.
    # sCu = [u_prefac*9.97938814e+01, u_prefac*9.97938814e+01]
    # sCp = [stage_prefac*2.06969312e+01, stage_prefac*7.69967729e+01]
    # sCv = [stage_prefac*1.55726136e-01, stage_prefac*5.42226523e-00]
    # sCen = 0.0
    # fCp = [final_prefac*3.82623819e+02, final_prefac*7.05315590e+03]
    # fCv = [final_prefac*5.89790058e+01, final_prefac*9.01459500e+01]
    # fCen = 0.0

    sCu = [.1, .1]
    sCp = [.1, .1]
    sCv = [0.01, 0.1]
    sCen = 0.0
    fCp = [100., 10.]
    fCv = [10., 1.]
    fCen = 0.0

    # stabilizaion cost par
    # u_prefac = 1.
    # stage_prefac = 1.
    # final_prefac = 1.
    # f_sCu = [u_prefac*9.97938814e+01, u_prefac*9.97938814e+01]
    # f_sCp = [stage_prefac*2.06969312e+01, stage_prefac*7.69967729e+01]
    # f_sCv = [stage_prefac*1.55726136e-01, stage_prefac*5.42226523e-00]
    # f_sCen = 0.0
    # f_fCp = [final_prefac*3.82623819e+02, final_prefac*7.05315590e+03]
    # f_fCv = [final_prefac*5.89790058e+01, final_prefac*9.01459500e+01]
    # f_fCen = 0.0
    f_sCu = [0.1, 0.1]
    f_sCp = [.1, .1]
    f_sCv = [.01, .01]
    f_sCen = 0.0
    f_fCp = [10., 10.]
    f_fCv = [1., 1.]
    f_fCen = 0.0


    # u_prefac = 1.
    # stage_prefac = 1.
    # final_prefac = 1.
    # sCu = [u_prefac*0.8220356078430472, u_prefac*0.8220356078430472]
    # sCp = [stage_prefac*0.6406768243361961, stage_prefac*0.5566465602921646]
    # sCv = [stage_prefac*0.13170941522322516, stage_prefac*0.036794663247905396]
    # sCen = 0.
    # fCp = [final_prefac*0.7170451397596873, final_prefac*0.7389953240562843]
    # fCv = [final_prefac*0.5243681881323512, final_prefac*0.39819013775238776]
    # fCen = 0.

    # tvlqr parameters
    # u_prefac = 1.0
    # sCu = [u_prefac*0.82, u_prefac*0.82]
    # sCp = [0.64, 0.56]
    # sCv = [0.13, 0.037]
    # sCen = 0.
    # fCp = [0.64, 0.56]
    # fCv = [0.13, 0.037]
    # fCen = 0.

if robot == "pendubot":
    # sCu = [0.2, 0.2]
    # sCp = [0.1, 0.2]
    # sCv = [0.1, 0.1]
    # sCen = 0.
    # fCp = [2500., 500.]
    # fCv = [500., 500.]
    # fCen = 0.

    sCu = [0.001, 0.001]
    sCp = [0.01, 0.01]
    sCv = [0.01, 0.01]
    sCen = 0.
    fCp = [100., 100.]
    fCv = [1., 1.]
    fCen = 0.

    # f_sCu = [0.0001, 0.0001]
    # f_sCp = [0.1, 0.1]
    # f_sCv = [0.01, 0.01]
    # f_sCen = 0.
    # f_fCp = [100., 1000.]
    # f_fCv = [1.0, 10.0]
    # f_fCen = 0.

    # f_sCu = [0.0001, 0.0001]
    # f_sCp = [0.1, 0.1]
    # f_sCv = [0.01, 0.01]
    # f_sCen = 0.
    # f_fCp = [10., 10.]
    # f_fCv = [0.1, 0.1]
    # f_fCen = 0.

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
controller.set_start(start)
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
controller.set_final_cost_parameters(sCu=f_sCu,
                                     sCp=f_sCp,
                                     sCv=f_sCv,
                                     sCen=f_sCen,
                                     fCp=f_fCp,
                                     fCv=f_fCv,
                                     fCen=f_fCen)
if init_csv_path is None:
    controller.compute_init_traj(N=N_init,
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
                                 integrator=integrator)
else:
    controller.load_init_traj(csv_path=init_csv_path,
                              num_break=40,
                              poly_degree=3)

controller.set_filter_args(filt=meas_noise_vfilter, x0=goal, dt=dt, plant=plant,
                           simulator=sim, velocity_cut=meas_noise_cut,
                           filter_kwargs=filter_kwargs)
if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)

controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   # imperfections=imperfections,
                                   plot_inittraj=True, plot_forecast=True,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   anim_dt=0.02)

# T, X, U = sim.simulate(t0=0.0, x0=start,
#                        tf=t_final, dt=dt, controller=controller,
#                        integrator="runge_kutta", imperfections=imperfections)

# saving and plotting

os.system(f"cp {init_csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

par_dict = {
            # "mass1": mass[0],
            # "mass2": mass[1],
            # "length1": length[0],
            # "length2": length[1],
            # "com1": com[0],
            # "com2": com[1],
            # "inertia1": inertia[0],
            # "inertia2": inertia[1],
            # "damping1": damping[0],
            # "damping2": damping[1],
            # "coulomb_friction1": cfric[0],
            # "coulomb_friction2": cfric[1],
            # "gravity": gravity,
            # "torque_limit1": torque_limit[0],
            # "torque_limit2": torque_limit[1],
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

#T_des, X_des, U_des = load_trajectory(init_csv_path)
T_des, X_des, U_des = load_trajectory(init_csv_path)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                T_des=T_des, X_des=X_des, U_des=U_des,
                save_to=os.path.join(save_dir, "timeseries"))
