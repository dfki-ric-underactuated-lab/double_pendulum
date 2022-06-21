import numpy as np

from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


# mass = [0.608, 0.630]
# length = [0.3, 0.2]
# com = [0.275, 0.166]
# damping = [0.081, 0.0]
# cfric = [0.093, 0.186]
# # cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.05472, 0.2522]
# torque_limit = [0.0, 1.0]
mass = [0.608, 0.63]
length = [0.3, 0.4]
com = [length[0], length[1]]
#damping = [0.081, 0.081]
damping = [0.0, 0.0]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2., mass[1]*length[1]**2.]
torque_limit = [0.0, 2.0]

dt = 0.002
t_final = 100.0

controller = LQRController(mass=mass,
                           length=length,
                           com=com,
                           damping=damping,
                           gravity=gravity,
                           coulomb_fric=cfric,
                           inertia=inertia,
                           torque_limit=torque_limit)

fric_comp = False
pos_pre = 0.1
vel_pre = 0.1
u_pre = 2.0

controller.set_goal([np.pi, 0., 0., 0.])
controller.set_cost_parameters(p1p1_cost=pos_pre*18.50,
                               p2p2_cost=pos_pre*15.90,
                               v1v1_cost=vel_pre*1.51,
                               v2v2_cost=vel_pre*1.66,
                               p1v1_cost=0.,
                               p1v2_cost=0.,
                               p2v1_cost=0.,
                               p2v2_cost=0.,
                               u1u1_cost=u_pre*1.,
                               u2u2_cost=u_pre*1.,
                               u1u2_cost=0.)
# controller.set_cost_parameters(p1p1_cost=11.67,
#                              p2p2_cost=3.87,
#                              v1v1_cost=0.10,
#                              v2v2_cost=0.11,
#                              p1v1_cost=0.,
#                              p1v2_cost=0.,
#                              p2v1_cost=0.,
#                              p2v2_cost=0.,
#                              u1u1_cost=0.18,
#                              u2u2_cost=0.18,
#                              u1u2_cost=0.)
# controller.set_cost_parameters(p1p1_cost=16.50,
#                                p2p2_cost=5.94,
#                                v1v1_cost=0.07,
#                                v2v2_cost=0.11,
#                                p1v1_cost=0.,
#                                p1v2_cost=0.,
#                                p2v1_cost=0.,
#                                p2v2_cost=0.,
#                                u1u1_cost=0.65,
#                                u2u2_cost=0.65,
#                                u1u2_cost=0.)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=200000.)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[8, 9],
               tau_limit=torque_limit,
               friction_compensation=fric_comp,
               #friction_terms=[0.093, 0.081, 0.186, 0.0],
               friction_terms=[0.093, 0.005, 0.15, 0.001],
               velocity_filter="lowpass",
               filter_args={"alpha": 0.3,
                            "kernel_size": 5,
                            "filter_size": 1},
               tau_filter="lowpass",
               tau_filter_args={"alpha": 0.5},
               save_dir="data/acrobot/tmotors/lqr_results")
