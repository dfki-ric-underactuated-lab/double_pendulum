import os
from datetime import datetime

from double_pendulum.controller.lqr.roa_paropt import roa_alternate_opt


init_pars = [1., 1., 1., 1., 1., 0.63, 0.3, 0.2]
par_prefactors = [20., 20., 10., 10., 10.,
                  1., 1., 1.]
bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
          [0.3, 1], [0.3, 0.5], [0.5, 1.]]
maxfevals_per_opt = 50
opt_order = ["d", "c"]
roa_backend = "najafi"
robot = "acrobot"
num_proc = 2


timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "lqr", "roa_altopt", timestamp)

best_par = roa_alternate_opt(init_pars=init_pars,
                             par_prefactors=par_prefactors,
                             bounds=bounds,
                             maxfevals_per_opt=maxfevals_per_opt,
                             opt_order=opt_order,
                             roa_backend=roa_backend,
                             robot=robot,
                             save_dir=save_dir,
                             num_proc=num_proc)
