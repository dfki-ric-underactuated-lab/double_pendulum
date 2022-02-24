import os
from datetime import datetime

from double_pendulum.controller.lqr.roa_paropt import roa_design_opt


lqr_pars = [1., 1., 1., 1., 1.]
init_pars = [0.63, 0.3, 0.2]
par_prefactors = [1., 1., 1.]
bounds = [[0.3, 1], [0.3, 0.5], [0.5, 1.]]
maxfevals = 29
roa_backend = "najafi"
robot = "acrobot"
num_proc = 2

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "lqr", "roa_design_paropt", timestamp)

best_par = roa_design_opt(lqr_pars=lqr_pars,
                          init_pars=init_pars,
                          par_prefactors=par_prefactors,
                          bounds=bounds,
                          maxfevals=maxfevals,
                          roa_backend=roa_backend,
                          robot=robot,
                          save_dir=save_dir,
                          num_proc=num_proc)
print(best_par)
