import os
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.utils.wrap_angles import wrap_angles_top,wrap_angles_diff
# from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.controller.SAC.SAC_controller import SACController
from double_pendulum.simulation.gym_env import (
   double_pendulum_dynamics_func,
)

## model parameters
robot = "pendubot"
# robot = "acrobot"

friction_compensation = True
stabilization = "lqr"

if robot == "pendubot":
    design = "design_C.1"
    model = "model_1.0"
    torque_limit = [5.0, 0.5]
    torque_limit_con = [5.0, 0.0]
    active_act = 0
    load_path = "lqr_data/design_C.1/model_1.1/pendubot/lqr"
    # gymnasium version
    model_path = "policies/design_C.1/model_1.0/pendubot/best_model/best_model.zip"

elif robot == "acrobot":
    design = "design_C.1"
    model = "model_1.0"
    torque_limit = [0.5, 5.0]
    active_act = 1
    load_path = "data_con/lqr_data/design_C.1/model_1.1/acrobot/lqr"

    # gymnasium version
    # model_path = "data_con/policies/design_C.1/model_1.0/acrobot/working_global_speed_penalty_1e5/best_model.zip"
    model_path = "data_con/policies/design_C.1/model_1.0/acrobot/works_very_well/best_model.zip"
			
## set model and controller parameters
model_par_path = "model_data/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)

mpar_con = model_parameters(filepath=model_par_path)


# the controller must react to the without-friction-compensation model
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit_con)

# measurement filter
meas_noise_cut = 0.1
meas_noise_vfilter = "lowpass"
filter_kwargs = {"lowpass_alpha": [1.0, 1.0, 0.2, 0.2]}

# control parameter
dt = 0.0025
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]
print("control frequency is", 1/dt)

# switching conditions between sac and lqr
lqr_pars = np.loadtxt(os.path.join(load_path, "controller_par.csv"))
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))

Q = np.diag(lqr_pars[:4])
R = np.diag([lqr_pars[4], lqr_pars[4]])
flag = False

def check_if_state_in_roa(S, rho, x):
    # print(x)
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    print(rad, rho)
    return rad < 1.0 * rho, rad

def condition1(t, x):
    return False

def condition2(t, x):
    y = wrap_angles_diff(x)
    flag,rad = check_if_state_in_roa(S,rho,y)
    print(t)
    print(y)
    # print(rad,rho)
    if flag:
        print(t)
        print(y)
        print(flag)
        return True
    else:
    	return False

## initialize sac controller
# initialize double pendulum dynamics
plant = SymbolicDoublePendulum(model_pars=mpar_con)
sim = Simulator(plant=plant)

dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
)

# initialize controller 1
controller1 = SACController(
    model_path = model_path,
    dynamics_func=dynamics_func,
    dt=dt,
    scaling=False
)

## initialize lqr controller
controller2 = LQRController(model_pars=mpar_con)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,
                          cost_to_go_cut=100000)

## initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
)

controller.set_filter_args(
    filt=meas_noise_vfilter, 
    velocity_cut=meas_noise_cut, 
    filter_kwargs=filter_kwargs
)

# setup friction compensation for combined controller
if friction_compensation:
    controller.set_friction_compensation(
        damping=mpar.b, coulomb_fric=[1.0 * mpar.cf[0], mpar.cf[1]]
    )
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[1, 2],
               tau_limit=torque_limit,
               save_dir=os.path.join("data_con", design, robot, "tmotors/sac_lqr_results")
               )





