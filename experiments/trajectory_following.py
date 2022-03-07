import asyncio

from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment as run_experiment_t
from double_pendulum.experiments.hardware_control_loop_mjbots import run_experiment as run_experiment_mj

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

if motors == "tmotors":
    run_experiment_t(controller=controller,
                     dt=dt,
                     t_final=t_final,
                     can_port="can0",
                     motor_ids=[8, 9],
                     tau_limit=torque_limit,
                     friction_compensation=True,
                     #friction_terms=[0.093, 0.081, 0.186, 0.0],
                     friction_terms=[0.093, 0.005, 0.15, 0.001],
                     velocity_filter="lowpass",
                     filter_args={"alpha": 0.2,
                                  "kernel_size": 5,
                                  "filter_size": 1},
                     save_dir="data/acrobot/tmotors/ilqr_results/traj_following")
elif motors == "mjbots":
    asyncio.run(run_experiment_mj(controller=controller,
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
