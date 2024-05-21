import os
import numpy as np

from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.controller.trajectory_following.trajectory_controller import (
    TrajectoryController,
)
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.filter.lowpass import lowpass_filter

design = "design_A.0"
traj_model = "model_2.1"
robot = "acrobot"

# model parameters
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
elif robot == "pendubot":
    torque_limit = [6.0, 0.0]
torque_limit_pid = [6.0, 6.0]
friction_compensation = True

# trajectory parameters
csv_path = os.path.join(
    "../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

T, X, U = load_trajectory(csv_path, True)
dt, t_final, _, _ = trajectory_properties(T, X)
t_final = t_final + 2.0

# swingup parameters
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.1

# controller parameters
Kp = 5.0
Ki = 1.0
Kd = 1.0


# switiching conditions
def condition1(t, x):
    return False


def condition2(t, x):
    goal = [np.pi, 0.0, 0.0, 0.0]
    eps = [0.2, 0.2, 2.0, 2.0]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.0:
        print(f"Stayed with TVLQR control in state x {x} at time {t}")
        return False
    else:
        print(f"Switched to PID control in state x {x} at time {t}")
        return True


# filter
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

controller1 = TrajectoryController(
    csv_path=csv_path, torque_limit=torque_limit, kK_stabilization=True
)
controller2 = PointPIDController(torque_limit=torque_limit_pid, dt=dt)
controller2.set_goal(goal)
controller2.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
)

controller.set_filter(filter)
if friction_compensation:
    controller.set_friction_compensation(
        damping=[0.15, 0.001], coulomb_fric=[0.093, 0.005]
    )
controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[8, 9],
    tau_limit=torque_limit_pid,
    save_dir=os.path.join("data", design, robot, "tmotors/ilqr_results/traj_following"),
)
