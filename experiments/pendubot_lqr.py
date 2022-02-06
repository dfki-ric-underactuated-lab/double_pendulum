import numpy as np

from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
# cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [3.0, 0.0]

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
controller.set_cost_parameters(p1p1_cost=0.01251931,
                               p2p2_cost=6.87772744,
                               v1v1_cost=6.51187283,
                               v2v2_cost=9.35785251,
                               p1v1_cost=0.,
                               p1v2_cost=0.,
                               p2v1_cost=0.,
                               p2v2_cost=0.,
                               u1u1_cost=1.02354949,
                               u2u2_cost=1.02354949,
                               u1u2_cost=0.)
# controller.set_cost_parameters(pp1_cost=10.0,
#                                pp2_cost=10.0,
#                                vv1_cost=1.,
#                                vv2_cost=1.,
#                                pv1_cost=0.,
#                                pv2_cost=0.,
#                                uu1_cost=2.0,
#                                uu2_cost=2.0)
controller.set_parameters(failure_value=0.0)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[8, 9],
               tau_limit=torque_limit,
               friction_compensation=False,
               friction_terms=[0.093, 0.186, 0.081, 0.0],
               velocity_filter="none",
               filter_args={"alpha": 0.3,
                            "kernel_size": 5,
                            "filter_size": 1},
               save_dir="data/pendubot/lqr_results")
