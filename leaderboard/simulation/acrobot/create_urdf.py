import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.utils.urdfs import generate_urdf

design = "design_C.0"
model = "model_3.0"
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
#print(mpar)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])


integrator = "runge_kutta"
dt = 0.002
t0 = 0.0
t_final = 10.0
x0 = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

mpar.save_dict("model_parameters.yml")
generate_urdf(urdf_in="../../../data/urdfs/design_A.0/model_1.0/acrobot.urdf",
              urdf_out="acrobot.urdf",
              model_pars=mpar)
