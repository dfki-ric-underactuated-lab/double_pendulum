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
max_torque = 3.0
max_velocity = 50.0
torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]

mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)


integrator = "runge_kutta"
dt = 0.002
t0 = 0.0
t_final = 10.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]
