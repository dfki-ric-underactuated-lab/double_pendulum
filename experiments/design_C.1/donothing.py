import os
import numpy as np

from double_pendulum.controller.trajectory_following.feed_forward import (
    FeedForwardController,
)
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


design = "design_C.1"
torque_limit = [0.0, 0.0]

# trajectory
dt = 0.005
t_final = 10.0
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N + 1)
u1 = np.zeros(N + 1)
u2 = np.zeros(N + 1)
U_des = np.array([u1, u2]).T

# controller
controller = FeedForwardController(
    T=T_des, U=U_des, torque_limit=[0.0, 0.0], num_break=40
)

controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[3, 1],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, "double-pendulum/donothing"),
    record_video=True,
)
