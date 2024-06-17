import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.filter.identity import identity_filter

# from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
# from double_pendulum.simulation.simulation import Simulator


design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"

torque_limit = [5.0, 5.0]
friction_compensation = True

if robot == "acrobot":
    torque_limit_con = [0.0, 5.0]
    active_act = 0
if robot == "pendubot":
    torque_limit_con = [5.0, 0.0]
    active_act = 1

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)

# mpar = model_parameters(filepath=model_par_path)
mpar_con = model_parameters(filepath=model_par_path)
# mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0.0, 0.0])
    mpar_con.set_cfric([0.0, 0.0])
mpar_con.set_torque_limit(torque_limit_con)

# motion parameters
goal = [np.pi, 0.0, 0.0, 0.0]
dt = 0.005
t_final = 10.0  # 4.985

# measurement filter
filter_velocity_cut = 0.1

# controller
N = 100
con_dt = dt
N_init = 100
max_iter = 20
max_iter_init = 1000
regu_init = 1.0
max_regu = 10000.0
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = False
shifting = 1
integrator = "runge_kutta"

if robot == "acrobot":
    sCu = [0.1, 0.1]
    sCp = [0.1, 0.1]
    sCv = [0.01, 0.01]
    sCen = 0.0
    fCp = [10.0, 10.0]
    fCv = [1.0, 1.0]
    fCen = 0.0

elif robot == "pendubot":
    sCu = [0.1, 0.1]
    sCp = [0.1, 0.1]
    sCv = [0.01, 0.01]
    sCen = 0.0
    fCp = [10.0, 10.0]
    fCv = [0.1, 0.1]
    fCen = 0.0

# filter
filter = identity_filter(filter_velocity_cut)

controller = ILQRMPCCPPController(model_pars=mpar_con)
controller.set_goal(goal)
controller.set_parameters(
    N=N,
    dt=con_dt,
    max_iter=max_iter,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    integrator=integrator,
    trajectory_stabilization=trajectory_stabilization,
    shifting=shifting,
)
controller.set_cost_parameters(
    sCu=sCu, sCp=sCp, sCv=sCv, sCen=sCen, fCp=fCp, fCv=fCv, fCen=fCen
)
controller.set_filter(filter)

if friction_compensation:
    controller.set_friction_compensation(
        damping=[0.001, 0.001], coulomb_fric=[0.09, 0.078]
    )

controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 8],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, robot, "tmotors/ilqr_mpc_stabi"),
)
