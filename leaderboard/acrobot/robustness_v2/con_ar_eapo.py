from pathlib import Path
from double_pendulum.controller.AR_EAPO import AR_EAPOController


name = "ar_eapo"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "AR-EAPO",
    "short_description": "Policy trained with average reward maximum entropy RL",
    "readme_path": f"readmes/{name}.md",
    "username": "rnilva",
}

model_path = Path(
    "../../../data/policies/design_C.1/model_1.1/acrobot/AR_EAPO/model.zip"
)
controller = AR_EAPOController(
    model_path=model_path,
    robot="acrobot",
    max_torque=6.0,
    max_velocity=20.0,
    deterministic=True,
)

controller.init()
