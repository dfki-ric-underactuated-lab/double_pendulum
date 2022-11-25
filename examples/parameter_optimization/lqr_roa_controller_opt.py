import os
from datetime import datetime

from double_pendulum.controller.lqr.roa.roa_paropt import roa_lqr_opt


model_pars = [0.63, 0.3, 0.2]
init_pars = [1., 1., 1., 1., 1.]
par_prefactors = [20., 20., 10., 10., 10.]
bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
maxfevals = 35
roa_backend = "najafi"
robot = "acrobot"
num_proc = 2

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "lqr", "roa_paropt", timestamp)

best_par = roa_lqr_opt(model_pars=model_pars,
                       init_pars=init_pars,
                       par_prefactors=par_prefactors,
                       bounds=bounds,
                       maxfevals=maxfevals,
                       roa_backend=roa_backend,
                       robot=robot,
                       save_dir=save_dir,
                       num_proc=num_proc)
print(best_par)
