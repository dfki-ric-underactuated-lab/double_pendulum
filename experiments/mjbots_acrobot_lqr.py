import numpy as np
import asyncio

from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.experiments.hardware_control_loop_mjbots import run_experiment


mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
# cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 3.0]

dt = 0.004
t_final = 200.0

controller = LQRController(mass=mass,
                           length=length,
                           com=com,
                           damping=damping,
                           gravity=gravity,
                           coulomb_fric=cfric,
                           inertia=inertia,
                           torque_limit=torque_limit)

controller.set_goal([np.pi, 0., 0., 0.])
controller.set_cost_parameters(p1p1_cost=11.67,
                               p2p2_cost=3.87,
                               v1v1_cost=0.05,  # 0.1
                               v2v2_cost=0.5,   # 0.11
                               p1v1_cost=0.,
                               p1v2_cost=0.,
                               p2v1_cost=0.,
                               p2v2_cost=0.,
                               u1u1_cost=3.0,  # 0.18
                               u2u2_cost=3.0,  # 0.18
                               u1u2_cost=0.)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=15.)
controller.init()

asyncio.run(run_experiment(controller=controller,
                           dt=dt,
                           t_final=t_final,
                           motor_ids=[1, 2],
                           tau_limit=torque_limit,
                           friction_compensation=False,
                           friction_terms=[0.0, 0.0, 0.0, 0.0],
                           velocity_filter="lowpass",
                           filter_args={"alpha": 0.15,
                                        "kernel_size": 21,
                                        "filter_size": 21},
                           save_dir="data/acrobot/mjbots/lqr_results"))
