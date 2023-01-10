import os
import time
from datetime import datetime
import yaml
import numpy as np
import multiprocessing

from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.controller.lqr.roa.ellipsoid import plotEllipse
from double_pendulum.controller.lqr.roa.roa_paropt import roa_modelpar_lossfunc
from double_pendulum.utils.optimization import (cma_optimization,
                                                scipy_par_optimization)


# interactive = False
num_proc = int(min(multiprocessing.cpu_count() - 1, 29))

# model parameters
robot = "pendubot"

mass = [0.6, 0.6]
length = [0.2, 0.3]
com = [0.2, 0.3]
damping = [0.0, 0.0]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2.0, mass[1]*length[1]**2.0]
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]

# simulation parameter
# dt = 0.005
# t_final = 6.0
# integrator = "runge_kutta"

# motion parameters
goal = [np.pi, 0, 0, 0]

# controller parameters
par_prefactors = np.asarray([1., 1., 1.])
bounds = [[0.6, 0.6001], [0.2, 0.3], [0.2, 0.3]]
init_pars = [0.6, 0.25, 0.25]

# roa parameters
roa_backend = "sos_con"

# optimization parameters
popsize_factor = 4
maxfevals = 1000
tolfun = 1e-11
tolx = 0.01
tolstagnation = 100

save_dir = os.path.join("../../results/design_optimization", robot, "lqr",
                        "roa_designopt")
os.makedirs(save_dir)

# loss function setup
loss_func = roa_modelpar_lossfunc(par_prefactors=par_prefactors,
                                        roa_backend=roa_backend,
                                        bounds=bounds,
                                        robot=robot)
loss_func.set_model_parameters(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)
loss_func.set_cost_parameters(p1p1_cost=1.,
                              p2p2_cost=1.,
                              v1v1_cost=1.,
                              v2v2_cost=1.,
                              p1p2_cost=0.,
                              v1v2_cost=0.,
                              p1v1_cost=0.,
                              p1v2_cost=0.,
                              p2v1_cost=0.,
                              p2v2_cost=0.,
                              u1u1_cost=1.,
                              u2u2_cost=1.,
                              u1u2_cost=0.)

# optimization
t0 = time.time()
best_par = cma_optimization(loss_func=loss_func,
                                init_pars=init_pars,
                                bounds=[0, 1],
                                save_dir=os.path.join(save_dir, "outcmaes"),
                                popsize_factor=popsize_factor,
                                maxfevals=maxfevals,
                                tolfun=tolfun,
                                tolx=tolx,
                                tolstagnation=tolstagnation,
                                num_proc=num_proc)
opt_time = (time.time() - t0) / 3600  # time in h

best_par = loss_func.rescale_pars(best_par)

print(best_par)

np.savetxt(os.path.join(save_dir, "model_par.csv"), best_par)
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
            # "dt": dt,
            # "t_final": t_final,
            # "integrator": integrator,
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3],
            "par_prefactors": par_prefactors.tolist(),
            "init_pars": init_pars,
            "bounds": bounds,
            "popsize_factor": popsize_factor,
            "maxfevals": maxfevals,
            "tolfun": tolfun,
            "tolx": tolx,
            "tolstagnation": tolstagnation
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)

# recalculate the roa for the best parameters and save plot
best_Q = np.diag((1., 1., 1., 1.))
best_R = np.diag((1., 1.))

design_params = {"m": [mass[0], best_par[0]],
                 "l": [best_par[1], best_par[2]],
                 "lc": [best_par[1], best_par[2]],
                 "b": damping,
                 "fc": cfric,
                 "g": gravity,
                 "I": [mass[0]*best_par[1]**2, best_par[0]*best_par[2]**2],
                 "tau_max": torque_limit}

roa_calc = caprr_coopt_interface(design_params=design_params,
                                 Q=best_Q,
                                 R=best_R,
                                 backend=roa_backend,
                                 robot=robot)
roa_calc._update_lqr(Q=best_Q, R=best_R)
vol, rho_f, S = roa_calc._estimate()

np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
np.savetxt(os.path.join(save_dir, "vol"), [vol])
# np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

plotEllipse(goal[0], goal[1], 0, 1, rho_f, S,
            save_to=os.path.join(save_dir, "roaplot"),
            show=False)
