import os
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.roa.roa_estimation import calc_roa

model_par_path = "../../data/system_identification/identified_parameters/design_C.1/model_1.1/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit([0.0, 5.0])

lqr_path = "../../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr"
lqr_pars = np.loadtxt(os.path.join(lqr_path, "controller_par.csv"))
Q = np.diag(lqr_pars[:4])
R = np.diag([lqr_pars[4], lqr_pars[4]])

calc_roa(
    model_par=mpar,
    goal=[np.pi, 0, 0, 0],
    Q=Q,
    R=R,
    roa_backend="sos",  # najafi, sos, sos_eq, prob
    najafi_evals=1000,
    robot="acrobot",
    save_dir="test_data/",
    plots=True,
    verbose=True,
)
