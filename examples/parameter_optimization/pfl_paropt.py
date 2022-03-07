import os
import time
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import SymbolicPFLController
from double_pendulum.utils.cmaes_controller_par_optimizer import (swingup_loss,
                                                                  cma_par_optimization,
                                                                  scipy_par_optimization)

interactive = True

# model parameter
robot = "acrobot"
with_cfric = False

mass = [0.608, 0.5]
length = [0.3, 0.4]
com = [length[0], length[1]]
#damping = [0.081, 0.0]
damping = [0.0, 0.0]
if with_cfric:
    cfric = [0.093, 0.186]
else:
    cfric = [0.0, 0.0]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

# simulation parameter
dt = 0.01
t_final = 5.00
x0 = [0.1, 0.0, 0.0, 0.0]
goal = np.array([np.pi, 0, 0, 0])
integrator = "runge_kutta"

# controller parameters
pfl_method = "collocated"

# optimization parameters
opt_method = "cma" #Nelder-Mead
bounds = np.asarray([[0., 20.],
                     [0., 20.],
                     [0., 20.]])
init_pars = [1.0, 1.0, 1.0]
loss_weights = [1.0, 0.0, 0.0] # state, tau smoothness, max_vel

popsize_factor = 4
maxfevals = 1000
tolfun = 1e-11
tolx = 1e-11
tolstagnation = 100
num_proc = 0

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, pfl_method, "paropt", timestamp)
os.makedirs(save_dir)

double_pendulum = SymbolicDoublePendulum(mass=mass,
                                         length=length,
                                         com=com,
                                         damping=damping,
                                         gravity=gravity,
                                         coulomb_fric=cfric,
                                         inertia=inertia,
                                         torque_limit=torque_limit)

sim = Simulator(plant=double_pendulum)

controller = SymbolicPFLController(mass,
                                   length,
                                   com,
                                   damping,
                                   gravity,
                                   cfric,
                                   inertia,
                                   torque_limit,
                                   robot,
                                   pfl_method)

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
    best_par = cma_par_optimization(loss_func=loss_func,
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
elif opt_method == "Nelder-Mead":
    best_par = scipy_par_optimization(loss_func=loss_func,
                                      init_pars=ipars,
                                      bounds=[[0, 1], [0, 1], [0, 1]],
                                      method=opt_method)
opt_time = (time.time() - t0) / 3600  # time in h

best_par = loss_func.rescale_pars(best_par)
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

if interactive:
    input("Press Enter to start simulation of best parameters")

    controller.set_cost_parameters(best_par[0], best_par[1], best_par[2])
    controller.init()

    T, X, U = sim.simulate_and_animate(t0=0.0,
                                       x0=x0,
                                       tf=t_final,
                                       dt=dt,
                                       controller=controller,
                                       integrator=integrator,
                                       phase_plot=False,
                                       save_video=False)
    plot_timeseries(T, X, U, None,
                    plot_energy=False,
                    pos_y_lines=[-np.pi, np.pi],
                    tau_y_lines=[-torque_limit[1], torque_limit[1]])
