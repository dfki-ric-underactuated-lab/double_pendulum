import numpy as np

from double_pendulum.controller.trajectory_following.feed_forward import FeedForwardController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


torque_limit = [8.0, 8.0]

# trajectory
dt = 0.002
t_final = 10.0
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N+1)
u1 = 0.4*np.sin(10.*T_des)
# u1 = np.zeros(N+1)
u2 = 0.8*np.cos(10.*T_des)
U_des = np.array([u1, u2]).T

# controller
controller = FeedForwardController(T=T_des,
                                   U=U_des,
                                   torque_limit=torque_limit,
                                   num_break=40)
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[8, 9],
               tau_limit=torque_limit,
               friction_compensation=False,
               friction_terms=None,
               velocity_filter="lowpass",
               filter_args={"alpha": 0.2,
                            "kernel_size": 5,
                            "filter_size": 1},
               save_dir="data/acrobot/tmotors/sysid")
