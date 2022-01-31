import numpy as np

from double_pendulum.controller.partial_feedback_linearization.pfl import EnergyShapingPFLAndLQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
#cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 5.0]

dt = 0.005
t_final = 100.0

controller = EnergyShapingPFLAndLQRController(mass,
                                              length,
                                              com,
                                              damping,
                                              gravity,
                                              cfric,
                                              inertia,
                                              torque_limit)
controller.set_goal([np.pi, 0., 0., 0.])

par = [6.97474837, 9.84031538, 9.1297417]
controller.set_hyperpar(kpos=par[0],
                        kvel=par[1],
                        ken=par[2])

controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               tau_limit=torque_limit,
               friction_compensation=False,
               save_dir="data/pfl_results")
