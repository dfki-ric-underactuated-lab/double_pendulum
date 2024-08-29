import numpy as np
import pandas as pd

from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.simulation.perturbations import (
    get_random_gauss_perturbation_array,
)


design = "design_C.1"

traj_csv_path = (
    "../../data/trajectories/design_C.1/model_1.1/acrobot/ilqr_1/trajectory.csv"
)

data = pd.read_csv(traj_csv_path)
time_traj = np.asarray(data["time"])
dt = 0.002  # time_traj[1] - time_traj[0]
t_final = time_traj[-1]

torque_limit = [8.0, 8.0]

controller = TrajPIDController(
    csv_path=traj_csv_path,
    use_feed_forward_torque=False,
    torque_limit=torque_limit,
)

controller.set_parameters(Kp=20.0, Ki=0.0, Kd=0.6)
controller.init()

perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
    t_final, dt, 3, 1.0, [0.05, 0.1], [0.5, 2.0]
)

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[3, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=torque_limit,
    save_dir="data/" + design + "/double-pendulum/fully_actuated_swingup",
    record_video=False,
    # perturbation_array=perturbation_array,
)
