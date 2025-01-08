import numpy as np
import copy

from double_pendulum.model.model_parameters import model_parameters

# np.random.seed(0)

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
mpar.set_torque_limit([0.0, 6.0])
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])

mpar_nolim = copy.copy(mpar)
mpar_nolim.set_torque_limit([6.0, 6.0])

integrator = "runge_kutta"
dt = 0.002
t0 = 0.0
t_final = 60.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

knockdown_after = 4.0
knockdown_length = 1.0
method = "height"
eps = [1e-2, 1e-2, 5e-1, 5e-1]
