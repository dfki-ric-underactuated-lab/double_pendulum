import numpy as np
import pandas as pd

from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment

design = "design_A.0"

# excitation_traj_csv = "../../data/system_identification/excitation_trajectories/trajectory-pos-20.csv"
excitation_traj_csv = "../../data/trajectories/acrobot/ilqr_v1.0/trajectory.csv"

data = pd.read_csv(excitation_traj_csv)
time_traj = np.asarray(data["time"])
dt = time_traj[1] - time_traj[0]
t_final = time_traj[-1]

print(dt)

torque_limit = [8.0, 8.0]
Kp = 20.0
Ki = 0.0
Kd = 0.6

controller = TrajPIDController(
    csv_path=excitation_traj_csv,
    use_feed_forward_torque=False,
    torque_limit=torque_limit,
)

controller.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 8],
    tau_limit=torque_limit,
    save_dir="data/" + design + "/double-pendulum/tmotors/sysid",
)
