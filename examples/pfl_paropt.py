import os
import shutil
import numpy as np
from datetime import datetime
from functools import partial

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import SymbolicPFLController
from double_pendulum.utils.cmaes_controller_par_optimizer import (swingup_loss,
                                                                  cma_par_optimization,
                                                                  scipy_par_optimization,
                                                                  swingup_test)

interactive = False

robot = "acrobot"
pfl_method = "collocated"

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]

if robot == "acrobot":
    torque_limit = [0.0, 4.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

dt = 0.01
t_final = 5.00
x0 = [0.1, 0.0, 0.0, 0.0]
goal = np.array([np.pi, 0, 0, 0])

kpos_pre = 1e1
kvel_pre = 1e1
ken_pre = 1e1

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, pfl_method, timestamp)
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
controller.set_lqr_parameters(failure_value=np.nan)

loss_func = partial(swingup_test,
                    simulator=sim,
                    controller=controller,
                    t_final=t_final,
                    dt=dt,
                    x0=x0,
                    integrator="runge_kutta",
                    goal=goal,
                    goal_accuracy=[0.1, 0.1, 0.2, 0.2],
                    par_prefactors=[kpos_pre, kvel_pre, ken_pre])
# kpos_pre=kpos_pre,
# kvel_pre=kvel_pre,
# ken_pre=ken_pre)


# loss_func = swingup_loss(simulator=sim,
#                          controller=controller,
#                          t_final=t_final,
#                          dt=dt,
#                          x0=x0,
#                          integrator="runge_kutta",
#                          goal=goal,
#                          goal_accuracy=[0.1, 0.1, 0.2, 0.2],
#                          par_prefactors=[kpos_pre, kvel_pre, ken_pre])

best_par = cma_par_optimization(loss_func=loss_func,
                                init_pars=[1.0, 1.0, 1.0],
                                bounds=[0, 1],
                                save_dir=os.path.join(save_dir, "outcmaes"))
# best_par = scipy_par_optimization(loss_func=loss_func,
#                                   init_pars=[1.0, 1.0, 1.0],
#                                   bounds=[[0, 1], [0, 1], [0, 1]],
#                                   method="Nelder-Mead")

best_par[0] *= kpos_pre
best_par[1] *= kvel_pre
best_par[2] *= ken_pre

print(best_par)

controller.set_cost_parameters(best_par[0], best_par[1], best_par[2])

if interactive:
    input("Press Enter to start simulation of best parameters")

    T, X, U = sim.simulate_and_animate(t0=0.0,
                                       x0=x0,
                                       tf=t_final,
                                       dt=dt,
                                       controller=controller,
                                       integrator="runge_kutta",
                                       phase_plot=False,
                                       save_video=False)
    plot_timeseries(T, X, U, None,
                    plot_energy=False,
                    pos_y_lines=[-np.pi, np.pi],
                    tau_y_lines=[-torque_limit[1], torque_limit[1]])
if interactive:
    save = input("Save parameters [Y/n]")
else:
    save = "yes"

if save in ["", "y", "Y", "yes", "Yes"]:
    np.savetxt(os.path.join(save_dir, "controller_par.csv"), best_par)
else:
    shutil.rmtree(save_dir)
