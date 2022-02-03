import numpy as np

from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


mass = [0.608, 0.630] #0.487 0.630-0.487=0.143
#mass = [0.608, 1.066]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
#cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 3.0]

dt = 0.004
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
controller.set_cost_parameters(pp1_cost=11.67,
                               pp2_cost=0.10,
                               vv1_cost=3.87,
                               vv2_cost=0.11,
                               pv1_cost=0.,
                               pv2_cost=0.,
                               uu1_cost=0.18,
                               uu2_cost=0.18)
# controller.set_cost_parameters(pp1_cost=50.0,
#                                pp2_cost=10.0,
#                                vv1_cost=10.0,
#                                vv2_cost=1.0,
#                                pv1_cost=0.,
#                                pv2_cost=0.,
#                                uu1_cost=1.0,
#                                uu2_cost=1.0)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15.)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids = [8, 9],
               tau_limit=torque_limit,
               friction_compensation=True,
               friction_terms=[0.093, 0.186, 0.081, 0.0],
               velocity_filter="lowpass",
               filter_args={"alpha": 0.2,
                            "kernel_size": 5,
                            "filter_size": 1},
               save_dir="data/acrobot/tmotors/lqr_results")
