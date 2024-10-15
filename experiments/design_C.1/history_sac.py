import os
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.controller.history_sac import HistorySACController
import numpy as np
from datetime import datetime
import pandas
from double_pendulum.analysis.leaderboard import leaderboard_scores
from double_pendulum.simulation.perturbations import (
    get_random_gauss_perturbation_array,
)


if __name__ == '__main__':

    # hier selbst einstellen
    model_id = 0 # index des Modells aus der models liste weiter unten
    lowpass = 0.0 # 0.0 ist kein lowpass, 0.7 - 0.95 normale range, Wert gibt an wie viel von alter action benutzt wird
    seed = None # aktiviert automatisch perturbations if not None
    print_score = True
    # controller.set_friction_compensation(damping=[0.001, 0.001], coulomb_fric=[0.16, 0.12])

    # Rest automatisch
    models = ['final_old', 'final_p', 'final_p_swing', 'final_p_b', 'final_p_b_swing', 'final_a', 'final_a_b']
    env_type = "pendubot"
    if model_id > 4:
        env_type = "acrobot"
    model = models[model_id]
    perturbations = None
    if seed is not None:
        np.random.seed(seed)
        perturbations, _, _, _ = get_random_gauss_perturbation_array(
            10.0, 0.002, 3, 1.0, [0.05, 0.1], [0.5, 0.6]
        )

    model_path = "../../data/policies/design_C.1/model_1.1/" + env_type + "/history_sac/" + model
    save_dir = os.path.join("data/history_sac/" + env_type + "/" + model)
    controller = HistorySACController(env_type, model_path=model_path, lowpass=lowpass)
    controller.init()

    # run experiment
    run_experiment(
        controller=controller,
        dt=0.002,
        t_final=10.0,
        can_port="can0",
        motor_ids=[3, 1],
        tau_limit=[6, 6],
        motor_directions=[1.0, -1.0],
        save_dir=save_dir,
        record_video=False,
        perturbation_array=perturbations
    )

    if print_score:
        subfolders = [f for f in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, f))]
        most_recent_folder = max(subfolders, key=lambda x: datetime.strptime(x, "%Y%m%d-%H%M%S"))
        most_recent_folder_path = os.path.join(save_dir, most_recent_folder)

        data_paths = {
            'History SAC': {
                'name': 'History SAC',
                'username': 'tfaust',
                'short_description': 'History SAC',
                'csv_path': [os.path.join(most_recent_folder_path, 'trajectory.csv')]
            }
        }

        from double_pendulum.model.model_parameters import model_parameters

        design = "design_C.1"
        model = "model_1.0"

        model_par_path = (
                "../../../data/system_identification/identified_parameters/"
                + design
                + "/"
                + model
                + "/model_parameters.yml"
        )
        mpar = model_parameters(filepath=model_par_path)

        leaderboard_scores(
            data_paths=data_paths,
            save_to=os.path.join(most_recent_folder_path, 'leaderboard.csv'),
            mpar=mpar,
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
                "tau_cost": 100.0,
                "tau_smoothness": 4.0,
                "velocity_cost": 400.0,
            },
            link_base="",
            simulation=False,
            score_version="v2",
        )
        df = pandas.read_csv(os.path.join(most_recent_folder_path, 'leaderboard.csv'))
        df = df.drop(df.columns[1], axis=1)
        print(
            df.sort_values(by=["Average RealAI Score"], ascending=False).to_markdown(
                index=False
            )
        )
