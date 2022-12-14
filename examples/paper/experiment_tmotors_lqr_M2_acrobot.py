import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


design = "design_C.0"
model = "model_3.0"
robot = "acrobot"

torque_limit = [5.0, 5.0]
friction_compensation = True

# model parameters
if robot == "pendubot":
    torque_limit_con = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit_con = [0.0, 5.0]
    active_act = 1

model_par_path = "../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)
#mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0., 0.])
    mpar_con.set_cfric([0., 0.])
mpar_con.set_torque_limit(torque_limit_con)

dt = 0.002
t_final = 30.0
goal = [np.pi, 0., 0., 0.]

# measurement filter
meas_noise_cut = 0.15
meas_noise_vfilter = "lowpass"
filter_kwargs = {"lowpass_alpha": [1., 1., 0.2, 0.2]}

# controller
Q = np.diag([0.68, 0.99, 0.42, 0.72])
R = np.diag((0.27, 0.27))
ctg_cut = 1000000

controller = LQRController(model_pars=mpar_con)
controller.set_goal(goal)
controller.set_cost_matrices(Q=Q, R=R)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=ctg_cut)
controller.set_filter_args(filt=meas_noise_vfilter,
         velocity_cut=meas_noise_cut,
         filter_kwargs=filter_kwargs)

if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[7, 8],
               tau_limit=torque_limit,
               save_dir=os.path.join("data", design, robot, "tmotors/lqr_results"))
