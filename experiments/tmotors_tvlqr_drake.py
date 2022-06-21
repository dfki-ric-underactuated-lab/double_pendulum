import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties

# model parameters
urdf_path = "../data/urdfs/acrobot.urdf"
robot = "acrobot"
cfric = [0., 0.]
motor_inertia = 0.
torque_limit = [0.0, 6.0]
torque_limit_pid = [6.0, 6.0]

# trajectory parameters
## tmotors v1.0
# csv_path = "../data/trajectories/acrobot/dircol/acrobot_tmotors_swingup_1000Hz.csv"
# read_with = "pandas"  # for dircol traj
# keys = "shoulder-elbow"

## tmotors v1.0
csv_path = "../data/trajectories/acrobot/ilqr_v1.0/trajectory.csv"
read_with = "numpy"
keys = ""

# tmotors v2.0
#csv_path = "../data/trajectories/acrobot/ilqr/trajectory.csv"
#read_with = "numpy"
#keys = ""

T, X, U = load_trajectory(csv_path, read_with, True, keys)
dt, t_final, _, _ = trajectory_properties(T, X)
#dt = 0.002
t_final = t_final + 2.
goal = [np.pi, 0., 0., 0.]

# controller parameters
Q = np.diag([0.64, 0.56, 0.13, 0.037])
R = np.eye(1)*0.82
Qf = np.copy(Q)

Kp = 5.
Ki = 1.0
Kd = 1.0

# switiching conditions
def condition1(t, x):
    return False

def condition2(t, x):
    goal = [np.pi, 0., 0., 0.]
    eps = [0.2, 0.2, 2.0, 2.0]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.:
        print(f"Stayed with TVLQR control in state x {x} at time {t}")
        return False
    else:
        print(f"Switched to PID control in state x {x} at time {t}")
        return True

# setup controller
controller1 = TVLQRController(
        csv_path=csv_path,
        urdf_path=urdf_path,
        read_with=read_with,
        torque_limit=torque_limit,
        robot=robot)
controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)
controller1.init()

controller2 = PointPIDController(
        torque_limit=torque_limit_pid,
        goal=goal,
        dt=dt)
controller2.set_parameters(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd)
controller2.init()

controller = CombinedController(
        controller1=controller1,
        controller2=controller2,
        condition1=condition1,
        condition2=condition2)

# run experiment
run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[8, 9],
               tau_limit=torque_limit_pid,
               friction_compensation=True,
               #friction_terms=[0.093, 0.081, 0.186, 0.0],
               #friction_terms=[0.093, 0.005, 0.15, 0.001],
               friction_terms=[0.0, 0.0, 0.15, 0.001],
               velocity_filter="lowpass",
               filter_args={"alpha": 0.2,
                            "kernel_size": 5,
                            "filter_size": 1},
               save_dir="data/acrobot/tmotors/tvlqr_drake_pid_results")
