import os
import argparse
import importlib
import numpy as np
import pandas

from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.analysis.leaderboard import leaderboard_scores
from double_pendulum.controller.global_policy_testing_controller import (
    GlobalPolicyTestingControllerV2,
)

from parameters import (
    mpar_nolim,
    t_final,
    goal,
    height,
    method,
    design,
    n_disturbances,
    reset_length,
    kp,
    ki,
    kd,
)


parser = argparse.ArgumentParser()
parser.add_argument("controller", help="name of the controller to simulate")
controller_arg = parser.parse_args().controller
if controller_arg[-3:] == ".py":
    controller_arg = controller_arg[:-3]

controller_name = controller_arg[4:]
imp = importlib.import_module(controller_arg)

# make sure the controller file has these four objects!
swingup_controller = imp.controller
dt = imp.dt
robot = imp.robot
conf = imp.leaderboard_config

global_policy_testing_controller = GlobalPolicyTestingControllerV2(
    swingup_controller,
    goal=goal,
    n_disturbances=n_disturbances,
    t_max=t_final,
    reset_length=reset_length,
    method=method,
    height=height,
    mpar=mpar_nolim,
    kp=kp,
    ki=ki,
    kd=kd,
    pos_limit=3.5 * np.pi,
    vel_limit=20.0,
    pid_pos_contribution_limit=[3.0, 3.0],
)

data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
save_dir = os.path.join("data", design, robot, controller_name)

run_experiment(
    controller=global_policy_testing_controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[2, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=[6.0, 6.0],
    save_dir=save_dir,
    record_video=True,
    safety_velocity_limit=25.0,
    safety_position_limit=4 * np.pi,
)

# os.chdir(search_dir)
run_directories = os.listdir(save_dir)
print("run_directories=", run_directories)
run_directories = [os.path.join(save_dir, d) for d in run_directories]
run_directories.sort(key=lambda x: os.path.getmtime(x))
save_dir_date = run_directories[-1]

save_lb_to = os.path.join(save_dir_date, "leaderboard_entry.csv")

conf["csv_path"] = os.path.join(save_dir_date, imp.leaderboard_config["csv_path"])
data_paths = {}
data_paths[imp.leaderboard_config["name"]] = conf

leaderboard_scores(
    data_paths=data_paths,
    save_to=save_lb_to,
    mpar=mpar_nolim,
    weights={
        # "swingup_time": 0.0,  # not used
        # "max_tau": 0.0,  # not used
        # "energy": 0.0,  # not used
        # "integ_tau": 0.0,  # not used
        # "tau_cost": 0.0,  # not used
        # "tau_smoothness": 0.0,  # not used
        # "velocity_cost": 0.0,  # not used
        "uptime": 1.0,
        # "n_swingups": 0.0,  # not used
    },
    normalize={
        # "swingup_time": 1.0,  # not used
        # "max_tau": 1.0,  # not used
        # "energy": 1.0,  # not used
        # "integ_tau": 1.0,  # not used
        # "tau_cost": 1.0,  # not used
        # "tau_smoothness": 1.0,  # not used
        # "velocity_cost": 1.0,  # not used
        "uptime": t_final,
        # "n_swingups": 1.0,  # not used
    },
    link_base="",
    score_version="v3",
)
df = pandas.read_csv(save_lb_to)
print(df.sort_values(by=["RealAI Score"], ascending=False).to_markdown(index=False))
