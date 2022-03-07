import os
import time
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.controller.ilqr.paropt import ilqrmpc_swingup_loss
from double_pendulum.utils.cmaes_controller_par_optimizer import (cma_par_optimization,
                                                                  scipy_par_optimization)


# interactive = False

# model parameters
robot = "acrobot"

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
# damping = [0.081, 0.0]
damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.02522]
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]

# simulation parameter
dt = 0.005
t_final = 6.0
integrator = "runge_kutta"

# controller parameters
N = 200
max_iter = 8
regu_init = 100
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6

bounds = [[0.001, 1.0],
          [0., 1.],
          [0., 1.],
          [0., 1.],
          [0., 1.],
          [0., 10000.],
          [0., 10000.],
          [0., 100.],
          [0., 100.]]

init_pars = [0.1, 0., 0., 0., 0., 100., 100., 10., 10.]

# swingup parameters
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0, 0, 0]
if robot == "acrobot":
    init_csv_path = "../data/acrobot/ilqr/trajopt/20220307-115818/trajectory.csv"

# optimization parameters
optimization_method = "cma"  # "Nelder-Mead"
loss_weights = [0.1, 0.0, 0.0, 0.9]
popsize_factor = 4
maxfevals = 1000
tolfun = 0.01
tolx = 0.01
tolstagnation = 100
num_proc = 0

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "ilqr", "mpc_paropt", timestamp)
os.makedirs(save_dir)

# loss object
loss_func = ilqrmpc_swingup_loss(bounds=bounds,
                                 loss_weights=loss_weights,
                                 start=start,
                                 goal=np.asarray(goal),
                                 csv_path=init_csv_path)

loss_func.set_model_parameters(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)
loss_func.set_parameters(N=N,
                         dt=dt,
                         max_iter=max_iter,
                         regu_init=regu_init,
                         max_regu=max_regu,
                         min_regu=min_regu,
                         break_cost_redu=break_cost_redu,
                         integrator=integrator)
loss_func.init()

# optimization
ipars = loss_func.unscale_pars(init_pars)
t0 = time.time()
if optimization_method == "cma":
    best_par = cma_par_optimization(
        loss_func=loss_func,
        init_pars=ipars,
        bounds=[0, 1],
        save_dir=os.path.join(save_dir, "outcmaes"),
        popsize_factor=popsize_factor,
        maxfevals=maxfevals,
        tolfun=tolfun,
        tolx=tolx,
        tolstagnation=tolstagnation,
        num_proc=num_proc)
else:
    best_par = scipy_par_optimization(loss_func=loss_func,
                                      init_pars=ipars,
                                      bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                                      method=optimization_method)
opt_time = (time.time() - t0) / 3600  # time in h

best_par = loss_func.rescale_pars(best_par)
print(best_par)

# saving
np.savetxt(os.path.join(save_dir, "controller_par.csv"), best_par)
np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])
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
            "max_iter": max_iter,
            "regu_init": regu_init,
            "max_regu": max_regu,
            "min_regu": min_regu,
            "break_cost_redu": break_cost_redu,
            "bounds": list(bounds),
            "optimization_method": optimization_method,
            "loss_weights": loss_weights,
            "popsize_factor": popsize_factor,
            "maxfevals": maxfevals,
            "tolfun": tolfun,
            "tolx": tolx,
            "tolstagnation": tolstagnation
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
