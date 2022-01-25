import numpy as np

from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop import run_experiment


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
#cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 1.0]

dt = 0.005
t_final = 100.0

controller = LQRController(mass=mass,
                           length=length,
                           com=com,
                           damping=damping,
                           gravity=gravity,
                           coulomb_fric=cfric,
                           inertia=inertia,
                           torque_limit=torque_limit)

controller.set_goal([np.pi, 0., 0., 0.])
controller.set_parameters(pp1_cost=2.0,
                          pp2_cost=2.0,
                          vv1_cost=0.1,
                          vv2_cost=0.1,
                          pv1_cost=0.,
                          pv2_cost=0.,
                          uu1_cost=2.0,
                          uu2_cost=2.0)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               tau_limit=torque_limit,
               friction_compensation=False,
               save_dir="data/lqr_results")
