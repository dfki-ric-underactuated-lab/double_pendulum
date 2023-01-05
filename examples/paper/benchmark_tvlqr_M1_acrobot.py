#!/usr/bin/python3

import os
from datetime import datetime
import numpy as np
import yaml
import pickle
import pprint

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.analysis.benchmark import benchmarker
from double_pendulum.analysis.utils import get_par_list
from double_pendulum.utils.csv_trajectory import load_trajectory


# model parameters
design = "design_A.0"
model = "model_2.1"
robot = "acrobot"

urdf_path = "../../data/urdfs/design_A.0/model_1.0/"+robot+".urdf"

if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]

model_par_path = "../../results/system_identification/identified_parameters/"+design+"/"+model[:-1]+"0"+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# init trajectory
init_csv_path = os.path.join("../../results/trajectories", design, model, robot, "ilqr/trajectory.csv")

# simulation parameter
T_des, X_des, U_des = load_trajectory(init_csv_path)
dt = T_des[1] - T_des[0]
t_final = T_des[-1]
#dt = 0.005
#t_final = 6.0  # 4.985
integrator = "runge_kutta"

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# acrobot good par
#Q = np.diag([0.64, 0.56, 0.13, 0.037])
#R = np.eye(1)*0.82
Q = np.diag([1.0, 1.0, 1.0, 1.0])
R = np.eye(1)
Qf = np.copy(Q)

# benchmark parameters
eps = [0.1, 0.1, 0.5, 0.5]
check_only_final_state = False

N_var = 21

compute_model_robustness = True
mpar_vars = ["Ir",
             "m1r1", "I1", "b1", "cf1",
             "m2r2", "m2", "I2", "b2", "cf2"]

Ir_var_list = np.linspace(0., 1e-4, N_var)
m1r1_var_list = get_par_list(mpar.m[0]*mpar.r[0], 0.75, 1.25, N_var)
I1_var_list = get_par_list(mpar.I[0], 0.75, 1.25, N_var)
b1_var_list = np.linspace(-0.1, 0.1, N_var)
cf1_var_list = np.linspace(-0.2, 0.2, N_var)
m2r2_var_list = get_par_list(mpar.m[1]*mpar.r[1], 0.75, 1.25, N_var)
m2_var_list = get_par_list(mpar.m[1], 0.75, 1.25, N_var)
I2_var_list = get_par_list(mpar.I[1], 0.75, 1.25, N_var)
b2_var_list = np.linspace(-0.1, 0.1, N_var)
cf2_var_list = np.linspace(-0.2, 0.2, N_var)

modelpar_var_lists = {"Ir": Ir_var_list,
                      "m1r1": m1r1_var_list,
                      "I1": I1_var_list,
                      "b1": b1_var_list,
                      "cf1": cf1_var_list,
                      "m2r2": m2r2_var_list,
                      "m2": m2_var_list,
                      "I2": I2_var_list,
                      "b2": b2_var_list,
                      "cf2": cf2_var_list}

compute_noise_robustness = True
meas_noise_mode = "vel"
meas_noise_sigma_list = np.linspace(0.0, 0.5, N_var)  # [0.0, 0.05, 0.1, 0.3, 0.5]
meas_noise_cut = 0.0
meas_noise_vfilters = ["None", "lowpass"]
meas_noise_vfilter_args = {"lowpass_alpha": [1., 1., 0.3, 0.3]}

compute_unoise_robustness = True
u_noise_sigma_list = np.linspace(0.0, 2.0, N_var)

compute_uresponsiveness_robustness = True
u_responses = np.linspace(0.1, 2.1, N_var)  # [1.0, 1.3, 1.5, 2.0]

compute_delay_robustness = True
delay_mode = "posvel"
delays = np.linspace(0.0, 0.04, N_var)  # [0.0, dt, 2*dt, 5*dt, 10*dt]

# create save directory
save_dir = os.path.join("../../results/benchmarks", design, model, robot, "tvlqr_drake")
os.makedirs(save_dir)

# construct simulation objects
controller = TVLQRController(csv_path=init_csv_path,
                             urdf_path=urdf_path,
                             model_pars=mpar,
                             torque_limit=torque_limit,
                             robot=robot,
                             save_dir=save_dir)
controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)
controller.init()

ben = benchmarker(controller=controller,
                  x0=start,
                  dt=dt,
                  t_final=t_final,
                  goal=goal,
                  epsilon=eps,
                  check_only_final_state=check_only_final_state,
                  integrator=integrator,
                  save_dir=save_dir)
ben.set_model_parameter(model_pars=mpar)
ben.set_init_traj(init_csv_path)
ben.set_cost_par(Q=Q, R=R, Qf=Qf)
ben.compute_ref_cost()
res = ben.benchmark(compute_model_robustness=compute_model_robustness,
                    compute_noise_robustness=compute_noise_robustness,
                    compute_unoise_robustness=compute_unoise_robustness,
                    compute_uresponsiveness_robustness=compute_uresponsiveness_robustness,
                    compute_delay_robustness=compute_delay_robustness,
                    mpar_vars=mpar_vars,
                    modelpar_var_lists=modelpar_var_lists,
                    meas_noise_mode=meas_noise_mode,
                    meas_noise_sigma_list=meas_noise_sigma_list,
                    meas_noise_cut=meas_noise_cut,
                    meas_noise_vfilters=meas_noise_vfilters,
                    meas_noise_vfilter_args=meas_noise_vfilter_args,
                    u_noise_sigma_list=u_noise_sigma_list,
                    u_responses=u_responses,
                    delay_mode=delay_mode,
                    delays=delays)
pprint.pprint(res)

# saving
f = open(os.path.join(save_dir, "results.pkl"), 'wb')
pickle.dump(res, f)
f.close()

os.system(f"cp {init_csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
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
            "sCu1": R[0],
            "sCu2": R[0],
            "sCp1": Q[0][0],
            "sCp2": Q[1][1],
            "sCv1": Q[2][2],
            "sCv2": Q[3][3],
            "fCp1": Qf[0][0],
            "fCp2": Qf[1][1],
            "fCv1": Qf[2][2],
            "fCv2": Qf[3][3],
            "epsilon": eps,
            "check_only_final_state": check_only_final_state
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
