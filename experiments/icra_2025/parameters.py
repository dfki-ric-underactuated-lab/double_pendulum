import numpy as np

from double_pendulum.model.model_parameters import model_parameters

# NOT allowed to change/overwrite these paramaters
# np.random.seed(0)
design = "design_C.1"
model = "model_1.0"

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar_nolim = model_parameters(filepath=model_par_path)
mpar_nolim.set_torque_limit([6.0, 6.0])

integrator = "runge_kutta"
t0 = 0.0
t_final = 60.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]
height = 0.9
method = "height"
kp = 4.0
ki = 0.0
kd = 1.0
n_disturbances = 10
reset_length = 0.5
