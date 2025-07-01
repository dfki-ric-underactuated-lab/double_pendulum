import os
import numpy as np
from datetime import datetime

from double_pendulum.controller.lqr.roa.roa_paropt import roa_lqr_opt
from double_pendulum.model.model_parameters import model_parameters

model_par_path = "../../data/system_identification/identified_parameters/design_C.1/model_1.1/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit([0.0, 5.0])

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", "acrobot", "lqr", "roa_paropt", timestamp)

c_par = roa_lqr_opt(
    model_par=mpar,
    goal=[np.pi, 0.0, 0.0, 0.0],
    init_pars=[1.0, 1.0, 1.0, 1.0, 1.0],
    par_prefactors=[20.0, 20.0, 10.0, 10.0, 10.0],
    bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    maxfevals=1000,
    sigma0=0.4,
    roa_backend="najafi",
    najafi_evals=1000,
    robot="acrobot",
    save_dir=save_dir,
    plots=True,
    num_proc=2,
    popsize_factor=4,
)

print(c_par)
