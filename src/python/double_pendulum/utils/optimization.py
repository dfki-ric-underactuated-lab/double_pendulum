from scipy import optimize
import cma
from cma.fitness_transformations import EvalParallel2
from scipy.optimize import minimize


# def solve_least_squares(Q, phi, xb0, bounds):
def solve_least_squares(loss_func, xb0, bounds, maxfevals):
    solve_opt = optimize.least_squares(fun=loss_func,
                                       x0=xb0[:],
                                       #args=(Q, phi),
                                       max_nfev=maxfevals,
                                       bounds=bounds)
    p1 = solve_opt.x
    success = solve_opt.success
    print("Least-Squares Optimization Success:", success)
    return p1

def cma_optimization(loss_func, init_pars, bounds,
                     save_dir="outcmaes/",
                     sigma0=0.4,
                     popsize_factor=3,
                     maxfevals=10000,
                     tolfun=1e-11,
                     tolx=1e-11,
                     tolstagnation=100,
                     num_proc=0):
    if save_dir[-1] != "/":
        sd = save_dir + "/"
    else:
        sd = save_dir

    if num_proc > 1:
        opts = cma.CMAOptions()
        opts.set("bounds", list(bounds))
        opts.set("verbose", -3)
        opts.set("popsize_factor", popsize_factor)
        opts.set("verb_filenameprefix", sd)
        opts.set("tolfun", tolfun)
        opts.set("tolx", tolx)
        opts.set("tolstagnation", tolstagnation)
        opts.set("maxfevals", maxfevals)
        es = cma.CMAEvolutionStrategy(init_pars,
                                      sigma0,
                                      opts)
        with EvalParallel2(loss_func, num_proc) as eval_all:
            while not es.stop():
                X = es.ask()
                es.tell(X, eval_all(X))
                es.disp()
                es.logger.add()  # doctest:+ELLIPSIS
    else:
        es = cma.CMAEvolutionStrategy(init_pars,
                                      sigma0,
                                      {'bounds': bounds,
                                       'verbose': -3,
                                       'popsize_factor': popsize_factor,
                                       'verb_filenameprefix': sd,
                                       'maxfevals': maxfevals,
                                       'tolfun': tolfun,
                                       'tolx': tolx,
                                       'tolstagnation': tolstagnation})

        es.optimize(loss_func)

    return es.result.xbest

def scipy_par_optimization(loss_func,
                           init_pars,
                           bounds,
                           method="Nelder-Mead",
                           maxfevals=10000):

    res = minimize(fun=loss_func,
                   x0=init_pars,
                   method=method,
                   bounds=bounds,
                   options={"maxiter": maxfevals,
                            "disp": True})

    return res.x
