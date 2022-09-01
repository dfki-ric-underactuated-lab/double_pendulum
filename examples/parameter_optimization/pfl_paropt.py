import os
import time
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import SymbolicPFLController
from double_pendulum.utils.cmaes_controller_par_optimizer import swingup_loss
from double_pendulum.utils.optimization import (cma_optimization,
                                                scipy_par_optimization)

design = "design_A.0"
model = "model_2.0"

robot = "pendubot"
pfl_method = "collocated"

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
dt = 0.001
t_final = 10.00
x0 = [0.1, 0.0, 0.0, 0.0]
goal = np.array([np.pi, 0, 0, 0])
integrator = "runge_kutta"

# optimization parameters
opt_method = "cma" #Nelder-Mead
bounds = np.asarray([[0., 10.],
                     [0., 10.],
                     [0., 10.]])
init_pars = [1., 1., 1.]
loss_weights = [1., 0.0, 0.] # state, tau smoothness, max_vel

popsize_factor = 6
maxfevals = 1000
tolfun = 1e-3
tolx = 1e-2
tolstagnation = 100
num_proc = 0

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "pfl", pfl_method, "paropt", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller = SymbolicPFLController(model_pars=mpar,
                                   robot=robot,
                                   pfl_method=pfl_method)
controller.set_goal(goal)

loss_func = swingup_loss(simulator=sim,
                         controller=controller,
                         t_final=t_final,
                         dt=dt,
                         x0=x0,
                         integrator=integrator,
                         goal=goal,
                         goal_accuracy=[0.1, 0.1, 0.2, 0.2],
                         bounds=bounds,
                         loss_weights=loss_weights)

ipars = loss_func.unscale_pars(init_pars)

t0 = time.time()
if opt_method == "cma":
    best_par = cma_optimization(loss_func=loss_func,
                                init_pars=ipars,
                                bounds=[0, 1],
                                save_dir=os.path.join(save_dir, "outcmaes"),
                                popsize_factor=popsize_factor,
                                maxfevals=maxfevals,
                                tolfun=tolfun,
                                tolx=tolx,
                                tolstagnation=tolstagnation,
                                num_proc=num_proc
                                )
else:
    best_par = scipy_par_optimization(loss_func=loss_func,
                                      init_pars=ipars,
                                      bounds=[[0, 1], [0, 1], [0, 1]],
                                      method=opt_method)
opt_time = (time.time() - t0) / 3600  # time in h

best_par = loss_func.rescale_pars(best_par)
print(best_par)

np.savetxt(os.path.join(save_dir, "controller_par.csv"), best_par)
np.savetxt(os.path.join(save_dir, "time.txt"), [opt_time])

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

par_dict = {
            "dt": dt,
            "t_final": t_final,
            "integrator": integrator,
            "start_pos1": x0[0],
            "start_pos2": x0[1],
            "start_vel1": x0[2],
            "start_vel2": x0[3],
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3],
            "opt_method": opt_method,
            "bounds": list(bounds),
            "init_pars": list(init_pars),
            "loss_weights": loss_weights,
            "popsize_factor": popsize_factor,
            "maxfevals": maxfevals,
            "tolfun": tolfun,
            "tolfx": tolx,
            "tolstagnation": tolstagnation,
            "num_proc": num_proc
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
