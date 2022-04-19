from scipy import optimize


def errfunc_with_friction(xb, Q, phi):
    return Q.flatten() - phi.dot(xb)


def solve_least_squares(Q, phi, xb0, bounds):
    solve_opt = optimize.least_squares(fun=errfunc_with_friction,
                                       x0=xb0[:],
                                       args=(Q, phi),
                                       max_nfev=100000000,
                                       bounds=bounds)
    p1 = solve_opt.x
    success = solve_opt.success
    print("Least-Squares Optimization Success:", success)
    return p1
