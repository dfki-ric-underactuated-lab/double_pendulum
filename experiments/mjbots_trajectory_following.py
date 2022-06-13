import asyncio

from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController
from double_pendulum.experiments.hardware_control_loop_mjbots import run_experiment

robot = "acrobot"
motors = "tmotors"

# model parameters
if robot == "acrobot":
    torque_limit = [0.0, 4.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

csv_path = "trajectory.csv"

dt = 0.005
t_final = 5.0

controller = TrajectoryController(csv_path=csv_path,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)
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
                           save_dir="data/acrobot/mjbots/ilqr_results/traj_following"))
