import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.prx_flow_matching.acrobot_flow_matching import AcrobotFlowMatchingController

MODEL_NAME_VS_MODEL_PATH = {
    "U-Net": "25_03_14-17_50_51_H_PADF_HIST_PADF_ATNT_LD1",
    "Transformer": "25_03_14-19_01_28_H_PADF_HIST_PADF_LD1_larger_transformer",
    "Transformer-Discounted": "25_03_14-21_02_37_H_PADF_HIST_PADF_LD0p99_transformer_large",
}

# MODEL_NAME = "U-Net"
# MODEL_NAME = "Transformer"
MODEL_NAME = "Transformer-Discounted"


name = "prx_acrobot_fm_trans_disc"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "prx_acrobot_fm_trans_disc",
    "short_description": "Flow matching.",
    "readme_path": f"readmes/{name}.md",
    "username": "garygra",
}

traj_model = "model_1.1"

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

controller = AcrobotFlowMatchingController(
    model_path=os.path.join(project_root, "trained_models", MODEL_NAME_VS_MODEL_PATH[MODEL_NAME]),
    horizon_length = 2
)

controller.init()
