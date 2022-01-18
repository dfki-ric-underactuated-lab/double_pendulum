import os
import numpy as np
import datetime
from functools import partial

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.pfl import EnergyShapingPFLAndLQRController
from double_pendulum.utils.cmaes_controller_par_optimizer import swingup_loss, cma_par_optimization, swingup_test


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 6.0]

goal = np.array([np.pi, 0, 0, 0])

double_pendulum = SymbolicDoublePendulum(mass=mass,
                                         length=length,
                                         com=com,
                                         damping=damping,
                                         gravity=gravity,
                                         coulomb_fric=cfric,
                                         inertia=inertia,
                                         torque_limit=torque_limit)


controller = EnergyShapingPFLAndLQRController(mass,
                                              length,
                                              com,
                                              damping,
                                              gravity,
                                              cfric,
                                              inertia,
                                              torque_limit)

sim = Simulator(plant=double_pendulum)

controller.set_goal(goal)

dt = 0.01
t_final = 5.00
x0 = [0.1, 0.0, 0.0, 0.0]

kpos_pre = 1e1
kvel_pre = 1e1
ken_pre = 1e1

loss_func = partial(swingup_test,
                    simulator=sim,
                    plant=double_pendulum,
                    controller=controller,
                    t_final=t_final,
                    dt=dt,
                    x0=x0,
                    integrator="runge_kutta",
                    goal=goal,
                    goal_accuracy=[0.1, 0.1, 0.2, 0.2],
                    kpos_pre=kpos_pre,
                    kvel_pre=kvel_pre,
                    ken_pre=ken_pre)


# loss_func = swingup_loss(simulator=sim,
#                          plant=double_pendulum,
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
                                bounds=[0, 1])

best_par[0] *= kpos_pre
best_par[1] *= kvel_pre
best_par[2] *= ken_pre

print(best_par)

controller.set_hyperpar(best_par[0], best_par[1], best_par[2])

input("Press Enter to start simulation of best parameters")

T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=x0,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator="runge_kutta",
                                   phase_plot=False,
                                   save_video=False)
save = input("Save parameters [Y/n]")

if save in ["", "y", "Y", "yes", "Yes"]:
    timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
    os.makedirs("data/"+timestamp)
    np.savetxt("data/"+timestamp+"/controller_par.csv", best_par)


