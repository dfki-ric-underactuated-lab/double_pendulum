import os
from datetime import datetime
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.saving import save_trajectory

robot = "acrobot"

# # model parameters
# mass = [0.608, 0.630]
# length = [0.3, 0.2]
# com = [0.275, 0.166]
# # damping = [0.081, 0.0]
# damping = [0.0, 0.0]
# # cfric = [0.093, 0.186]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.05472, 0.02522]
# torque_limit = [0.0, 4.0]

# model parameters
mass = [0.608, 0.5]
length = [0.3, 0.4]
com = [length[0], length[1]]
# damping = [0.081, 0.0]
damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
if robot == "acrobot":
    torque_limit = [0.0, 4.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

# simulation parameter
dt = 0.005
t_final = 6.0
integrator = "runge_kutta"

# controller parameters
N = 100
N_init = 1000
max_iter = 5
max_iter_init = 1000
regu_init = 100
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6

# acrobot good par
sCu = [9.97938814e-02, 9.97938814e-02]
sCp = [2.06969312e-02, 7.69967729e-02]
sCv = [1.55726136e-04, 5.42226523e-03]
sCen = 0.0
fCp = [3.82623819e+02, 7.05315590e+03]
fCv = [5.89790058e+01, 9.01459500e+01]
fCen = 0.0

# sCu = [9.97938814e-02, 9.97938814e-02]
# sCp = [2.06969312e+02, 7.69967729e+02]
# sCv = [1.55726136e+01, 5.42226523e+01]
# sCen = 0.0
# fCp = [0., 0.]
# fCv = [0., 0.]
# fCen = 0.0

# sCu = [9.96090757e-02, 9.96090757e-02]
# sCp = [2.55362809e-02, 9.65397113e-02]
# sCv = [2.17121720e-05, 6.80616778e-03]
# sCen = 0.0
# fCp = [2.56167942e+02, 7.31751057e+03]
# fCv = [9.88563736e+01, 9.67149494e+01]
# fCen = 0.0

# sCu = [0.01, 0.01]
# sCp = [0., 0.]
# sCv = [0.0, 0.0]
# sCen = 0.
# fCp = [300., 10000.]
# fCv = [500., 100.]
# fCen = 0.

# sCu = [9.64008003e-04, 3.69465206e-04]
# sCp = [9.00160028e-04, 8.52634075e-04]
# sCv = [3.62146682e-03, 3.49079107e-02]
# sCen = 1.08953921e-05
# fCp = [9.88671633e+02, 0.0]
# fCv = [6.75242351e+00, 9.95354381e+00]
# fCen = 6.36798375e+00

# init trajectory
latest_dir = sorted(os.listdir(os.path.join("data", robot, "ilqr", "trajopt")))[-1]
init_csv_path = os.path.join("data", robot, "ilqr", "trajopt", latest_dir, "trajectory.csv")

# init_sCu = [9.64008003e-04, 3.69465206e-04]
# init_sCp = [9.00160028e-04, 8.52634075e-04]
# init_sCv = [3.62146682e-05, 3.49079107e-04]
# init_sCen = 1.08953921e-05
# init_fCp = [9.88671633e+03, 7.27311031e+03]
# init_fCv = [6.75242351e+01, 9.95354381e+01]
# init_fCen = 6.36798375e+01

# init_sCu = [0.01, 0.01]
# init_sCp = [0.0, 0.0]
# init_sCv = [0.0, 0.0]
# init_sCen = 0.0
# init_fCp = [300, 10000]
# init_fCv = [500, 100]
# init_fCen = 0.0

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "ilqr", "mpc", timestamp)
os.makedirs(save_dir)

# construct simulation objects
plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = ILQRMPCCPPController(mass=mass,
                                  length=length,
                                  com=com,
                                  damping=damping,
                                  gravity=gravity,
                                  coulomb_fric=cfric,
                                  inertia=inertia,
                                  torque_limit=torque_limit)

controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(N=N,
                          dt=dt,
                          max_iter=max_iter,
                          regu_init=regu_init,
                          max_regu=max_regu,
                          min_regu=min_regu,
                          break_cost_redu=break_cost_redu,
                          integrator=integrator)
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
controller.load_init_traj(csv_path=init_csv_path)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   plot_inittraj=True, plot_forecast=True,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"))

# T, X, U = sim.simulate(t0=0.0, x0=start,
#                        tf=t_final, dt=dt, controller=controller,
#                        integrator="runge_kutta")

# saving and plotting

os.system(f"cp {init_csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))

par_dict = {"mass1": mass[0],
            "mass2": mass[1],
            "length1": length[0],
            "length2": length[1],
            "com1": com[0],
            "com2": com[1],
            "inertia1": inertia[0],
            "inertia2": inertia[1],
            "damping1": damping[0],
            "damping2": damping[1],
            "coulomb_friction1": cfric[0],
            "coulomb_friction2": cfric[1],
            "gravity": gravity,
            "torque_limit1": torque_limit[0],
            "torque_limit2": torque_limit[1],
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

trajectory = np.loadtxt(init_csv_path, skiprows=1, delimiter=",")
T_des = trajectory.T[0]
pos1_des = trajectory.T[1]
pos2_des = trajectory.T[2]
vel1_des = trajectory.T[3]
vel2_des = trajectory.T[4]
tau1_des = trajectory.T[5]
tau2_des = trajectory.T[6]

U_des = np.vstack((tau1_des, tau2_des)).T
X_des = np.vstack((pos1_des, pos2_des, vel1_des, vel2_des)).T

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                T_des=T_des, X_des=X_des, U_des=U_des,
                save_to=os.path.join(save_dir, "timeseries"))
