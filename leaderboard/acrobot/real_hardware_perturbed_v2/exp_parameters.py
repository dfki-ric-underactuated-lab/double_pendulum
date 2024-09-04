import numpy as np

from double_pendulum.model.model_parameters import model_parameters

design = "design_C.1"
model = "model_1.0"
robot = "acrobot"

model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

dt = 0.002
t0 = 0.0
t_final = 10.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

perturbations_per_joint = 2
min_time_distance = 1.0
perturbation_duration = [0.05, 0.1]
perturbation_amplitudes = [0.5, 0.75]
