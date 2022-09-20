import os
from datetime import datetime
import numpy as np
import yaml
import pickle
import pprint

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.analysis.benchmark import benchmarker
from double_pendulum.analysis.utils import get_par_list

design = "design_A.0"
model = "model_2.1"
robot = "acrobot"

# model parameters
# damping = [0., 0.]
cfric = [0., 0.]
motor_inertia = 0.0  # 8.8e-5
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model[:-1]+"0"+"/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
mpar.set_cfric(cfric)
# mpar.set_damping(damping)
mpar.set_torque_limit(torque_limit)

# simulation parameter
dt = 0.005
t_final = 4.985
integrator = "runge_kutta"

# swingup parameters
start = [np.pi+0.05, -0.2, 0.0, 0.0]
goal = [np.pi, 0., 0., 0.]

# controller parameters

# acrobot good par
Q = np.diag((0.97, 0.93, 0.39, 0.26))
R = np.diag((0.11, 0.11))
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
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "lqr", "benchmark", timestamp)
os.makedirs(save_dir)

# construct simulation objects
controller = LQRController(model_pars=mpar)
controller.set_goal([np.pi, 0., 0., 0.])
controller.set_cost_parameters(p1p1_cost=Q[0, 0],
                               p2p2_cost=Q[1, 1],
                               v1v1_cost=Q[2, 2],
                               v2v2_cost=Q[3, 3],
                               p1v1_cost=0.,
                               p1v2_cost=0.,
                               p2v1_cost=0.,
                               p2v2_cost=0.,
                               u1u1_cost=R[0, 0],
                               u2u2_cost=R[1, 1],
                               u1u2_cost=0.)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1e9)
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
            "fCv2": Qf[3][3]
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
