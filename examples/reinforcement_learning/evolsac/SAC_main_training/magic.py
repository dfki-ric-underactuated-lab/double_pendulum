import os
import shutil

import numpy as np
import pandas
from double_pendulum.analysis.leaderboard import leaderboard_scores
from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from double_pendulum.model.model_parameters import model_parameters
from sim_controller import simulate_controller
from stable_baselines3.common.callbacks import BaseCallback


def load_controller(dynamics_func, model, window_size, include_time):
    name = "evolsac"
    leaderboard_config = {
        "csv_path": name + "/sim_swingup.csv",
        "name": name,
        "simple_name": name,
        "short_description": "SAC finetuning for both swingup and stabilisation",
        "readme_path": f"readmes/{name}.md",
        "username": "MarcoCali0",
    }
    controller = EvolSACController(
        model=model,
        dynamics_func=dynamics_func,
        window_size=window_size,
        include_time=include_time,
    )
    controller.init()
    return controller, leaderboard_config


def magic_score(
    dynamics_func,
    model,
    folder,
    folder_id,
    window_size,
    max_torque,
    include_time,
    index=None,
):
    design = "design_C.1"
    integrator = "runge_kutta"
    dt = 0.002
    t0 = 0.0
    t_final = 10.0
    x0 = [0.0, 0.0, 0.0, 0.0]
    goal = [np.pi, 0.0, 0.0, 0.0]

    model_par_path = (
        f"../../../../data/system_identification/identified_parameters/"
        + design
        + "/"
        + "model_1.0"
        + "/model_parameters.yml"
    )
    torque_limit = [0.0, max_torque] if folder == "acrobot" else [max_torque, 0.0]
    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit)

    controller, leaderboard_config = load_controller(
        dynamics_func, model, window_size, include_time
    )

    data_dir = f"data_{folder}/{folder_id}"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    controller_name = f"evolsac"
    save_dir = (
        f"{data_dir}/{controller_name}"
        if index is None
        else f"{data_dir}/{controller_name}/{index}"
    )
    if index is not None:
        leaderboard_config["csv_path"] = f"{controller_name}/{index}/sim_swingup.csv"
    save_to = os.path.join(save_dir, "leaderboard_entry.csv")

    simulate_controller(
        controller, save_dir, mpar, dt, t_final, t0, x0, goal, integrator
    )

    conf = leaderboard_config

    conf["csv_path"] = os.path.join(data_dir, leaderboard_config["csv_path"])
    data_paths = {}
    data_paths[leaderboard_config["name"]] = conf

    leaderboard_scores(
        data_paths=data_paths,
        save_to=save_to,
        mpar=mpar,
        # weights={"swingup_time": 0.5, "max_tau": 0.1, "energy": 0.0, "integ_tau": 0.4, "tau_cost": 0.0, "tau_smoothness": 0.0},
        weights={
            "swingup_time": 1.0,
            "max_tau": 0.0,
            "energy": 1.0,
            "integ_tau": 0.0,
            "tau_cost": 1.0,
            "tau_smoothness": 1.0,
            "velocity_cost": 1.0,
        },
        normalize={
            "swingup_time": 20.0,
            "max_tau": 1.0,  # not used
            "energy": 60.0,
            "integ_tau": 1.0,  # not used
            "tau_cost": 20.0,
            "tau_smoothness": 0.1,
            "velocity_cost": 400,
        },
        link_base="",
        score_version="v2",
    )
    df = pandas.read_csv(save_to)
    score = np.array(df["RealAI Score"])[0]
    print("RealAI Score = ", score)
    return score


def copy_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dest_file = os.path.join(dest_folder, filename)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)


class MagicCallback(BaseCallback):
    def __init__(
        self,
        path,
        folder_id,
        dynamics_func,
        robot,
        window_size,
        max_torque,
        include_time,
    ):
        super().__init__(False)
        self.path = f"{path}{folder_id}"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.best = -np.inf
        self.folder_id = folder_id
        self.dynamics_func = dynamics_func
        self.robot = robot
        self.window_size = window_size
        self.max_torque = max_torque
        self.include_time = include_time

    def _on_step(self) -> bool:
        score = magic_score(
            self.dynamics_func,
            self.model,
            self.robot,
            self.folder_id,
            self.window_size,
            self.max_torque,
            self.include_time,
        )
        if score >= self.best:
            self.best = score
            self.model.save(f"{self.path}/best_model")
            copy_files(f"./data_{self.robot}/{self.folder_id}/evolsac/", self.path)
        return True


import concurrent.futures


class BruteMagicCallback(BaseCallback):
    def __init__(
        self,
        path,
        folder_id,
        dynamics_func,
        robot,
        window_size,
        max_torque,
        include_time,
    ):
        super().__init__(False)
        self.path = f"{path}{folder_id}"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        self.folder_id = folder_id
        self.dynamics_func = dynamics_func
        self.robot = robot
        self.window_size = window_size
        self.max_torque = max_torque
        self.include_time = include_time
        self.iteration = 0
        self.executor = concurrent.futures.ThreadPoolExecutor()

    def _on_step(self) -> bool:
        self.iteration += 1
        self.executor.submit(lambda: async_store(self.iteration, self))
        return True


import tempfile

from stable_baselines3 import SAC


def deepcopy_model(model):
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "temp_model")
        model.save(model_path)
        copied_model = SAC.load(model_path)
    return copied_model


def async_store(iteration, callback: BruteMagicCallback):
    model = deepcopy_model(callback.model)
    _ = magic_score(
        callback.dynamics_func,
        model,
        callback.robot,
        callback.folder_id,
        callback.window_size,
        callback.max_torque,
        callback.include_time,
        index=iteration,
    )
    if not os.path.exists(f"{callback.path}/{iteration}"):
        os.makedirs(f"{callback.path}/{iteration}", exist_ok=True)
    model.save(f"{callback.path}/{iteration}/best_model")
    copy_files(
        f"./data/{callback.folder_id}/evolsac/{iteration}",
        f"{os.path.join(callback.path, str(iteration))}",
    )
