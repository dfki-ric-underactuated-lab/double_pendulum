import numpy as np

from double_pendulum.controller.trajectory_following.feed_forward import (
    FeedForwardController,
)

name = "donothing"
leaderboard_config = {
    "csv_path": "trajectory.csv",
    "name": name,
    "simple_name": "donothing",
    "short_description": "This controller does nothing.",
    "readme_path": f"readmes/{name}.md",
    "username": "fwiebe",
}

robot = "double_pendulum"

# trajectory
dt = 0.005
t_final = 60.0
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
