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

## model parameters
design = "design_C.0"
model = "model_3.0"
traj_model = "model_3.1"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# simulation parameter
dt = 0.005
t_final = 10.  # 5.985
integrator = "runge_kutta"
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# controller parameters
#N = 20
N = 100
con_dt = dt
N_init = 1000
max_iter = 10
#max_iter = 100
max_iter_init = 1000
regu_init = 1.
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = True
shifting = 1

init_csv_path = os.path.join("../../data/trajectories", design, traj_model, robot, "ilqr_2/trajectory.csv")

if robot == "acrobot":
    sCu = [.1, .1]
    sCp = [.1, .1]
    sCv = [0.01, 0.1]
    sCen = 0.0
    fCp = [100., 10.]
    fCv = [10., 1.]
    fCen = 0.0

    f_sCu = [0.1, 0.1]
    f_sCp = [.1, .1]
    f_sCv = [.01, .01]
    f_sCen = 0.0
    f_fCp = [10., 10.]
    f_fCv = [1., 1.]
    f_fCen = 0.0

    # sCu = [9.979, 9.979]
    # sCp = [20.7, 77.0]
    # sCv = [0.16, 5.42]
    # sCen = 0.0
    # fCp = [382.62, 7053.16]
    # fCv = [58.98, 90.15]
    # fCen = 0.0

    # f_sCu = sCu
    # f_sCp = sCp
    # f_sCv = sCv
    # f_sCen = sCen
    # f_fCp = fCp
    # f_fCv = fCv
    # f_fCen = fCen

if robot == "pendubot":

    sCu = [0.001, 0.001]
    sCp = [0.01, 0.01]
    sCv = [0.01, 0.01]
    sCen = 0.
    fCp = [100., 100.]
    fCv = [1., 1.]
    fCen = 0.

    f_sCu = sCu
    f_sCp = sCp
    f_sCv = sCv
    f_sCen = sCen
    f_fCp = fCp
    f_fCv = fCv
    f_fCen = fCen

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "ilqr", "mpc", timestamp)
os.makedirs(save_dir)

# construct simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = ILQRMPCCPPController(model_pars=mpar)
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
controller.load_init_traj(csv_path=init_csv_path,
                          num_break=40,
                          poly_degree=3)

controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=True, plot_forecast=True,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   anim_dt=0.02)

# saving and plotting

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
os.system(f"cp {init_csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
controller.save(save_dir)
save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

T_des, X_des, U_des = load_trajectory(init_csv_path)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                T_des=T_des, X_des=X_des, U_des=U_des,
                save_to=os.path.join(save_dir, "timeseries"))
