import os
import time
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.ilqr.paropt import ilqrmpc_swingup_loss
from double_pendulum.utils.optimization import (cma_optimization,
                                                scipy_par_optimization)

design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"

# model parameter
if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_cfric([0.0, 0.0])
mpar.set_damping([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

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
init_csv_path = os.path.join("../../data/trajectories/", design, traj_model, robot, "ilqr_1/trajectory.csv")

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
save_dir = os.path.join("data", design, model, robot, "ilqr", "mpc_paropt", timestamp)
os.makedirs(save_dir)

# loss object
loss_func = ilqrmpc_swingup_loss(bounds=bounds,
                                 loss_weights=loss_weights,
                                 start=start,
                                 goal=np.asarray(goal),
                                 csv_path=init_csv_path)

loss_func.set_model_parameters(model_pars=mpar)
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
    best_par = cma_optimization(
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

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

par_dict = {
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
