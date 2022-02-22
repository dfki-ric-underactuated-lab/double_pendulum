import os
import numpy as np
from datetime import datetime
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.trajectory_optimization.ilqr.ilqr_cpp import ilqr_calculator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.saving import save_trajectory
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController

# model parameters
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
# damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.02522]
torque_limit = [0.0, 6.0]

# simulation parameter
dt = 0.005
t_final = 5.0
integrator = "runge_kutta"

# controller parameters
N = 1000
max_iter = 1000
regu_init = 100
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6

# looking good
# [9.97938814e-02 2.06969312e-02 7.69967729e-02 1.55726136e-04
#  5.42226523e-03 3.82623819e+02 7.05315590e+03 5.89790058e+01
#  9.01459500e+01]
sCu = [9.97938814e-02, 9.97938814e-02]
sCp = [2.06969312e-02, 7.69967729e-02]
sCv = [1.55726136e-04, 5.42226523e-03]
sCen = 0.0
fCp = [3.82623819e+02, 7.05315590e+03]
fCv = [5.89790058e+01, 9.01459500e+01]
fCen = 0.0

# looking good
# [9.96090757e-02 2.55362809e-02 9.65397113e-02 2.17121720e-05
#  6.80616778e-03 2.56167942e+02 7.31751057e+03 9.88563736e+01
#  9.67149494e+01]
# sCu = [9.96090757e-02, 9.96090757e-02]
# sCp = [2.55362809e-02, 9.65397113e-02]
# sCv = [2.17121720e-05, 6.80616778e-03]
# sCen = 0.0
# fCp = [2.56167942e+02, 7.31751057e+03]
# fCv = [9.88563736e+01, 9.67149494e+01]
# fCen = 0.0

# sCu = [0.2, 0.2]
# sCp = [0.1, 0.2]
# sCv = [0.2, 0.2]
# sCen = 0.0
# fCp = [1500., 500.]
# fCv = [10., 10.]
# fCen = 0.

# [9.64008003e-04 3.69465206e-04 9.00160028e-04 8.52634075e-04
#  3.62146682e-05 3.49079107e-04 1.08953921e-05 9.88671633e+03
#  7.27311031e+03 6.75242351e+01 9.95354381e+01 6.36798375e+01]

# [6.83275883e-01 8.98205799e-03 5.94690881e-04 2.60169706e-03
# 4.84307636e-03 7.78152311e-03 2.69548072e+02 9.99254272e+03
# 8.55215256e+02 2.50563565e+02 2.57191000e+01]


# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

il = ilqr_calculator()
il.set_model_parameters(mass=mass,
                        length=length,
                        com=com,
                        damping=damping,
                        gravity=gravity,
                        coulomb_fric=cfric,
                        inertia=inertia,
                        torque_limit=torque_limit)
il.set_parameters(N=N,
                  dt=dt,
                  max_iter=max_iter,
                  regu_init=regu_init,
                  max_regu=max_regu,
                  min_regu=min_regu,
                  break_cost_redu=break_cost_redu,
                  integrator=integrator)
il.set_cost_parameters(sCu=sCu,
                       sCp=sCp,
                       sCv=sCv,
                       sCen=sCen,
                       fCp=fCp,
                       fCv=fCv,
                       fCen=fCen)
il.set_start(start)
il.set_goal(goal)

# computing the trajectory
T, X, U = il.compute_trajectory()

# saving

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", "acrobot", "ilqr", "trajopt", timestamp)
os.makedirs(save_dir)

traj_file = os.path.join(save_dir, "trajectory.csv")

il.save_trajectory_csv()
os.system("mv trajectory.csv " + traj_file)
# save_trajectory(filename=filename,
#                 T=T, X=X, U=U)

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
            "max_iter": max_iter,
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

# plotting
U = np.append(U, [[0.0, 0.0]], axis=0)
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                save_to=os.path.join(save_dir, "timeseries"))

# simulation
plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=traj_file,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator, phase_plot=False,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   plot_forecast=True)
