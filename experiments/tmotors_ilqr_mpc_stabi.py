import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters


torque_limit = [5.0, 5.0]
robot = "acrobot"
friction_compensation = True

if robot == "acrobot":
    torque_limit_con = [0.0, 5.0]
    active_act = 0
if robot == "pendubot":
    torque_limit_con = [5.0, 0.0]
    active_act = 1

model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
# model_par_path = "../data/system_identification/identified_parameters/tmotors_v2.0/model_parameters_est.yml"

# mpar = model_parameters(filepath=model_par_path)
mpar_con = model_parameters(filepath=model_par_path)
#mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0., 0.])
    mpar_con.set_cfric([0., 0.])
mpar_con.set_torque_limit(torque_limit_con)

# motion parameters
goal = [np.pi, 0., 0., 0.]
dt = 0.005
t_final = 10.0  # 4.985

# measurement filter
meas_noise_cut = 0.
meas_noise_vfilter = "none"
filter_kwargs = {"lowpass_alpha": [1., 1., 0.3, 0.3]}

# controller
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
integrator = "runge_kutta"

if robot == "acrobot":
    sCu = [0.1, 0.1]
    sCp = [.1, .1]
    sCv = [.01, .01]
    sCen = 0.0
    fCp = [10., 10.]
    fCv = [1., 1.]
    fCen = 0.0

elif robot == "pendubot":
    sCu = [0.1, 0.1]
    sCp = [0.1, 0.1]
    sCv = [0.01, 0.01]
    sCen = 0.
    fCp = [10., 10.]
    fCv = [0.1, .1]
    fCen = 0.

controller = ILQRMPCCPPController(model_pars=mpar_con)
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

controller.set_filter_args(filt=meas_noise_vfilter,
         velocity_cut=meas_noise_cut,
         filter_kwargs=filter_kwargs)

if friction_compensation:
    controller.set_friction_compensation(damping=[0.001, 0.001], coulomb_fric=[0.09, 0.078])

controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[7, 8],
               tau_limit=torque_limit,
               save_dir="data/"+robot+"/tmotors/ilqr_mpc_stabi")
