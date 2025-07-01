import os
from datetime import datetime
import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.roa.roa_paropt import roa_alternate_opt

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", "acrobot", "lqr", "roa_altopt", timestamp)

best_par = roa_alternate_opt(
    model_par=model_parameters(),
    goal=[np.pi, 0.0, 0.0, 0.0],
    init_pars=[1.0, 1.0, 1.0, 1.0, 1.0, 0.63, 0.3, 0.2],
    par_prefactors=[20.0, 20.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    bounds=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0.3, 1], [0.3, 0.5], [0.5, 1.0]],
    maxfevals_per_opt=50,
    opt_order=["d", "c"],
    roa_backend="najafi",
    robot="acrobot",
    save_dir=save_dir,
    num_proc=50,
)
