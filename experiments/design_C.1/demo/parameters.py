import numpy as np

dt = 0.003
t_final = 60.0
integrator = "runge_kutta"
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0, 0, 0]
torque_limit = [6.0, 6.0]

design = "design_C.1"
robot = "acrobot"
model = "model_1.0"

model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
