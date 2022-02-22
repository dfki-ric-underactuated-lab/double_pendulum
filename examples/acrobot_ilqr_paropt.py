import os
import time
import numpy as np
from datetime import datetime
import yaml

from double_pendulum.trajectory_optimization.ilqr.paropt import ilqr_trajopt_loss
from double_pendulum.utils.cmaes_controller_par_optimizer import (cma_par_optimization,
                                                                  scipy_par_optimization)


interactive = False

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
t_final = 5.0
integrator = "runge_kutta"

# controller parameters
N = 1000
max_iter = 1000
regu_init = 100
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6

par_prefactors = [0.1,
                  0.1, 0.1,
                  0.1, 0.1,
                  10000, 10000,
                  100, 100]

# swingup parameters
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0, 0, 0]

# optimization parameters
loss_weights = [1.0, 0.0]
popsize_factor = 4
maxfevals = 10
tolfun = 0.01
tolx = 0.01
tolstagnation = 100


timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "ilqr", "trajopt_paropt", timestamp)
os.makedirs(save_dir)

# loss function setup
loss_func = ilqr_trajopt_loss(par_prefactors=par_prefactors,
                              loss_weights=loss_weights,
                              start=start,
                              goal=np.asarray(goal))

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

# optimization

t0 = time.time()

best_par = cma_par_optimization(
    loss_func=loss_func,
    init_pars=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    bounds=[0, 1],
    save_dir=os.path.join(save_dir, "outcmaes"),
    popsize_factor=popsize_factor,
    maxfevals=maxfevals,
    tolfun=tolfun,
    tolx=tolx,
    tolstagnation=tolstagnation)

# best_par = scipy_par_optimization(loss_func=loss_func,
#                                   init_pars=[1.0, 1.0, 1.0],
#                                   bounds=[[0, 1], [0, 1], [0, 1]],
#                                   method="Nelder-Mead")

opt_time = (time.time() - t0) / 3600  # time in h

best_par = np.asarray(best_par)*np.asarray(par_prefactors)
print(best_par)

np.savetxt(os.path.join(save_dir, "controller_par.csv"), best_par)
np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

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
            "par_prefactors": par_prefactors,
            "loss_weights": loss_weights,
            "popsize_factor": popsize_factor,
            "maxfevals": maxfevals,
            "tolfun": tolfun,
            "tolx": tolx,
            "tolstagnation": tolstagnation
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
