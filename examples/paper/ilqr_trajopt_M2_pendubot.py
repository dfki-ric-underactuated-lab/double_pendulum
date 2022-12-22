import time
from datetime import datetime
import os
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.trajectory_optimization.ilqr.ilqr_cpp import ilqr_calculator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController

## model parameters
design = "design_C.0"
model = "model_3.0"
robot = "pendubot"

if robot == "acrobot":
    torque_limit = [0.0, 5.0]
if robot == "pendubot":
    torque_limit = [5.0, 0.0]

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# controller parameters
N = 6000
max_iter = 100000
regu_init = 100.
max_regu = 1000000.
min_regu = 0.01
break_cost_redu = 1e-6

# simulation parameter
dt = 0.001
t_final = N*dt
integrator = "runge_kutta"

if robot == "acrobot":
    sCu = [9.97938814e+01, 9.97938814e+01]
    sCp = [2.06969312e+01, 7.69967729e+01]
    sCv = [1.55726136e-01, 5.42226523e-00]
    sCen = 0.0
    fCp = [3.82623819e+02, 7.05315590e+03]
    fCv = [5.89790058e+01, 9.01459500e+01]
    fCen = 0.0
if robot == "pendubot":
    sCu = [1., 1.]
    sCp = [0.1, 0.2]
    sCv = [0., 0.]
    sCen = 0.
    fCp = [10., 20.]
    fCv = [10., 20.]
    fCen = 0.

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

t0 = time.time()
il = ilqr_calculator()
il.set_model_parameters(model_pars=mpar)
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
print("Computing time: ", time.time() - t0, "s")

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "ilqr", "trajopt", timestamp)
os.makedirs(save_dir)

traj_file = os.path.join(save_dir, "trajectory.csv")

il.save_trajectory_csv()
os.system("mv trajectory.csv " + traj_file)
# save_trajectory(csv_path=filename,
#                 T=T, X=X, U=U)

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

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

# plotting
U = np.append(U, [[0.0, 0.0]], axis=0)
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                save_to=os.path.join(save_dir, "timeseries"))

# simulation
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=traj_file,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator, phase_plot=False,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"),
                                   plot_inittraj=True)
