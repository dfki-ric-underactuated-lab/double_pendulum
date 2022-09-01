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

design = "design_A.0"
model = "model_2.0"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 4.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# simulation parameter
dt = 0.005
t_final = 10.0
integrator = "runge_kutta"

process_noise_sigmas = [0., 0., 0., 0.]
meas_noise_sigmas = [0., 0., 0.05, 0.05]
meas_noise_cut = 0.0
meas_noise_vfilter = "none"
meas_noise_vfilter_args = {"alpha": [1., 1., 0.3, 0.3]}
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0., 0.]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# controller parameters
N = 200
N_init = 1000
max_iter = 2
max_iter_init = 100
regu_init = 1.
max_regu = 10000.
min_regu = 0.0001
break_cost_redu = 1e-6
trajectory_stabilization = False

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

if robot == "acrobot":
    # works with high velocities (40-50 rad/s)
    # f_sCu = [0.0001, 0.0001]
    # f_sCp = [.1, .1]
    # f_sCv = [.01, .01]
    # f_sCen = 0.0
    # f_fCp = [10., 10.]
    # f_fCv = [1., 1.]
    # f_fCen = 0.0

    f_sCu = [0.0001, 0.0001]
    f_sCp = [.1, .1]
    f_sCv = [.01, .5]
    f_sCen = 0.0
    f_fCp = [10., 10.]
    f_fCv = [1., 1.]
    f_fCen = 1.0

if robot == "pendubot":
    f_sCu = [0.0001, 0.0001]
    f_sCp = [0.1, 0.2]
    f_sCv = [0., 0.]
    f_sCen = 0.
    f_fCp = [10., 10.]
    f_fCv = [1., 1.]
    f_fCen = 1.

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "ilqr", "mpc_free", timestamp)
os.makedirs(save_dir)

# construct simulation objects
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

controller = ILQRMPCCPPController(model_pars=mpar)
controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(N=N,
                          dt=dt,
                          max_iter=max_iter,
                          regu_init=regu_init,
                          max_regu=max_regu,
                          min_regu=min_regu,
                          break_cost_redu=break_cost_redu,
                          integrator=integrator,
                          trajectory_stabilization=trajectory_stabilization)
controller.set_cost_parameters(sCu=f_sCu,
                               sCp=f_sCp,
                               sCv=f_sCv,
                               sCen=f_sCen,
                               fCp=f_fCp,
                               fCv=f_fCv,
                               fCen=f_fCen)
controller.compute_init_traj(N=N_init,
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
                             integrator=integrator)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=False, plot_forecast=True,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   anim_dt=0.02)

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
            "sCu1": f_sCu[0],
            "sCu2": f_sCu[1],
            "sCp1": f_sCp[0],
            "sCp2": f_sCp[1],
            "sCv1": f_sCv[0],
            "sCv2": f_sCv[1],
            "sCen": f_sCen,
            "fCp1": f_fCp[0],
            "fCp2": f_fCp[1],
            "fCv1": f_fCv[0],
            "fCv2": f_fCv[1],
            "fCen": f_fCen
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)

save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

X_meas = sim.meas_x_values

plot_timeseries(T, X, U, None,
                plot_energy=False,
                #T_des=T[1:],
                #X_des=X_filt,
                X_meas=X_meas,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                save_to=os.path.join(save_dir, "timeseries"))
