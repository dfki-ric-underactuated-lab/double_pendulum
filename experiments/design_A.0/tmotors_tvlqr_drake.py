import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties
from double_pendulum.filter.lowpass import lowpass_filter


# model parameters
design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"
urdf_path = "../data/urdfs/" + robot + ".urdf"

save_dir = os.path.join("data", design, robot, "tmotors/tvlqr_drake_pid_results")

friction_compensation = True

torque_limit = [0.0, 6.0]
torque_limit_pid = [6.0, 6.0]

model_par_path = (
    "../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)
# mpar.set_motor_inertia(0.)
if friction_compensation:
    mpar.set_damping([0.0, 0.0])
    mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# trajectory parameters
csv_path = os.path.join(
    "../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

T, X, U = load_trajectory(csv_path, True)
dt, t_final, _, _ = trajectory_properties(T, X)
# dt = 0.002
t_final = t_final + 2.0

# swingup parameters
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

# filter args
lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.15

# controller parameters
Q = np.diag([0.64, 0.56, 0.13, 0.037])
R = np.eye(1) * 0.82
Qf = np.copy(Q)

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

# controller
controller1 = TVLQRController(
    csv_path=csv_path,
    urdf_path=urdf_path,
    model_pars=mpar,
    torque_limit=torque_limit,
    robot=robot,
    save_dir=save_dir,
)
controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller2 = PointPIDController(torque_limit=torque_limit_pid, dt=dt)
controller2.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller2.set_goal(goal)

controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
)
controller.set_filter(filter)

if friction_compensation:
    controller.set_friction_compensation(
        damping=[0.0, 0.001], coulomb_fric=[0.0, 0.078]
    )

controller.init()

# run experiment
run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 8],
    tau_limit=torque_limit_pid,
    save_dir=save_dir,
)
