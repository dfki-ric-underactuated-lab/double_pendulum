import os
import yaml
import numpy as np
from pydrake.solvers import MathematicalProgram, Solve

# from pydrake.solvers.csdp import CsdpSolver
from pydrake.symbolic import Variables
import pydrake.symbolic as sym

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.pydrake_symbolic_plant import SymbolicDoublePendulumPydrake
from double_pendulum.controller.lqr.roa.ellipsoid import (
    quadForm,
    sampleFromEllipsoid,
    volEllipsoid,
    plotEllipse,
)
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.lqr.roa.roa_check import (
    lqr_check_isnotNaN,
)


def estimate_roa_najafi(plant, controller, goal, S, n):
    rho = 10.0
    for i in range(n):
        # sample initial state from sublevel set
        # check if it fullfills Lyapunov conditions
        x_bar = sampleFromEllipsoid(S, rho)
        x = goal + x_bar

        tau = controller.get_control_output(x)

        xdot = plant.rhs(0, x, tau)

        V = quadForm(S, x_bar)

        Vdot = 2 * np.dot(x_bar, np.dot(S, xdot))

        if V < rho and Vdot > 0.0:
            # if one of the lyapunov conditions is not satisfied
            rho = V

    return rho


# def estimate_roa_najafi_direct(plant, controller, goal, S, n):
#
#     rho = 10.0
#     for i in range(n):
#         # sample initial state from sublevel set
#         # check if it fullfills Lyapunov conditions
#         x_bar = sampleFromEllipsoid(S, rho)
#         x = goal + x_bar
#
#         tau = controller.get_control_output(x)
#
#         xdot = plant.rhs(0, x, tau)
#
#         V = quadForm(S, x_bar)
#
#         Vdot = 2 * np.dot(x_bar, np.dot(S, xdot))
#
#         if V > rho:
#             print("something is fishy")
#         # V < rho is true trivially, because we sample from the ellipsoid
#         if Vdot > 0.0:
#             # if one of the lyapunov conditions is not satisfied
#             rho = V
#
#     return rho


class estimate_roa_probabilistic:
    """
    class for probabilistic RoA estimation for linear (or linearized) systems
    under (infinite horizon) TILQR control.

    Takes a configuration dict and requires passing a callback function in
    which the simulation is done.  The callback function returns the result of
    the simulation as a boolean (True = Success)

    the conf dict has the following structure

        roaConf={   "x0Star": <goal state to stabilize around>, "xBar0Max":
        <bounds that define the first (over) estimate of the RoA to sample
        from>, "S": <cost to go matrix TODO change this to V for other
        stabilizing controllers> "nSimulations": <number of simulations> }

    TODO: generalize for non LQR systems -> V instead of S
    """

    def __init__(self, roaConf, simFct):
        self.x0Star = roaConf["x0Star"]
        self.xBar0Max = roaConf["xBar0Max"]
        self.S = roaConf["S"]
        self.nSims = roaConf["nSimulations"]
        self.simClbk = simFct

        self.rhoHist = []
        self.simSuccessHist = []

        rho0 = quadForm(self.S, self.xBar0Max)
        self.rhoHist.append(rho0)

    def doEstimate(self):
        for sim in range(self.nSims):
            # sample initial state from previously estimated RoA
            x0Bar = sampleFromEllipsoid(self.S, self.rhoHist[-1])
            JStar0 = quadForm(self.S, x0Bar)  # calculate cost to go
            x0 = self.x0Star + x0Bar  # error to absolute coords

            simSuccess = self.simClbk(x0)

            if not simSuccess:
                self.rhoHist.append(JStar0)
            else:
                self.rhoHist.append(self.rhoHist[-1])

            self.simSuccessHist.append(simSuccess)

        return self.rhoHist, self.simSuccessHist


def estimate_roa_sos(
    model_par,
    goal,
    rho,
    S,
    K,
    robot,
    taylor_deg=3,
    lambda_deg=4,
    mode=2,
    verbose=False,
):
    """
    params      --> parameters of pendulum and controller
    taylor_deg  --> degree of the taylor approximation
    lamda_deg   --> degree of SOS lagrange multipliers

    It solves the feasibility SOS problem in one of the three modes:
    0: completely unconstrained (no actuation limits)
        -->     fastest, it will theoretically overestimate the actual RoA
    1(TODO): check only where controls are not saturated
        -->     mid    , will underestimate actual RoA.
        We could actually use a for this.
        Does it make sense to maximize the region in which no saturation
        occurs, i.e. the linearization is valid and the params are well?
    2: also check for saturated dynamics
        -->     slowest, but best estimate. Still much more fast wrt the najafi method.
    """

    # LQR parameters
    # S = S  # params["S"]
    # K = K  # params["K"]

    active_motor_ind = None
    if robot == "acrobot":
        active_motor_ind = 1
    elif robot == "pendubot":
        active_motor_ind = 0

    plant = SymbolicDoublePendulumPydrake(model_pars=model_par)

    # Saturation parameters
    u_plus = np.array(model_par.tl[active_motor_ind])
    u_minus = -np.array(model_par.tl[active_motor_ind])

    # Opt. Problem definition (Indeterminates in error coordinates)
    prog = MathematicalProgram()
    x_bar = prog.NewIndeterminates(4, "x_bar")
    x_bar_1 = x_bar[0]
    x_bar_2 = x_bar[1]
    xd_bar_1 = x_bar[2]
    xd_bar_2 = x_bar[3]

    # Dynamics definition
    x_star = np.array([np.pi, 0, 0, 0])  # desired state in physical coordinates
    x = x_star + x_bar  # state in physical coordinates
    u = -K.dot(x_bar)  # control input

    acc = plant.forward_dynamics(x, u)
    acc_minus = plant.forward_dynamics(x, u_plus)
    acc_plus = plant.forward_dynamics(x, u_minus)

    # Manipulator eqs (this is not in error coords)
    # q1 = x[0]
    # q2 = x[1]
    qd1 = x[2]
    qd2 = x[3]

    # Taylor approximation
    env = {x_bar_1: 0.0, x_bar_2: 0.0, xd_bar_1: 0.0, xd_bar_2: 0.0}
    acc1_approx = sym.TaylorExpand(f=acc[0], a=env, order=taylor_deg)
    acc2_approx = sym.TaylorExpand(f=acc[1], a=env, order=taylor_deg)
    f = np.array([[qd1], [qd2], [acc1_approx], [acc2_approx]])
    # Definition of the Lyapunov function and of its derivative
    V = x_bar.dot(S.dot(x_bar))
    Vdot = (V.Jacobian(x_bar).dot(f))[0]

    Vdot_check = 2 * np.dot(
        x_bar, np.dot(S, f.flatten())
    )  # Checking the effect of the approximation on Vdot
    f_true = np.array([qd1, qd2, acc[0], acc[1]])
    Vdot_true = 2 * np.dot(x_bar, np.dot(S, f_true.flatten()))

    # Multipliers definition
    lambda_b = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()

    if mode == 2:
        acc1_approx_minus = sym.TaylorExpand(f=acc_minus[0], a=env, order=taylor_deg)
        acc2_approx_minus = sym.TaylorExpand(f=acc_minus[1], a=env, order=taylor_deg)
        acc1_approx_plus = sym.TaylorExpand(f=acc_plus[0], a=env, order=taylor_deg)
        acc2_approx_plus = sym.TaylorExpand(f=acc_plus[1], a=env, order=taylor_deg)
        f_minus = np.array([[qd1], [qd2], [acc1_approx_minus], [acc2_approx_minus]])
        f_plus = np.array([[qd1], [qd2], [acc1_approx_plus], [acc2_approx_plus]])

        Vdot_minus = (V.Jacobian(x_bar).dot(f_minus))[0]
        Vdot_plus = (V.Jacobian(x_bar).dot(f_plus))[0]

        # u in linear range
        lambda_2 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        lambda_3 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        # uplus
        lambda_c = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        lambda_4 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        # uminus
        lambda_a = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
        lambda_1 = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()

    # Completely unconstrained dynamics
    epsilon = 10e-20
    if mode == 0:
        prog.AddSosConstraint(-Vdot + lambda_b * (V - rho) - epsilon * x_bar.dot(x_bar))

    # Considering the input saturation in the constraints
    if mode == 2:
        if robot == "acrobot":
            nom1 = (+K.dot(x_bar) + u_minus)[
                1
            ]  # where both nom1 and nom2 are < 0, the nominal dynamics have to be fullfilled
            nom2 = (-K.dot(x_bar) - u_plus)[1]
            neg = (-K.dot(x_bar) - u_minus)[
                1
            ]  # where this is < 0, the negative saturated dynamics have to be fullfilled
            pos = (+K.dot(x_bar) + u_plus)[
                1
            ]  # where this is < 0, the positive saturated dynamics have to be fullfilled
        if robot == "pendubot":
            nom1 = (+K.dot(x_bar) + u_minus)[
                0
            ]  # where both nom1 and nom2 are < 0, the nominal dynamics have to be fullfilled
            nom2 = (-K.dot(x_bar) - u_plus)[0]
            neg = (-K.dot(x_bar) - u_minus)[
                0
            ]  # where this is < 0, the negative saturated dynamics have to be fullfilled
            pos = (+K.dot(x_bar) + u_plus)[
                0
            ]  # where this is < 0, the positive saturated dynamics have to be fullfilled

        prog.AddSosConstraint(
            -Vdot
            + lambda_b * (V - rho)
            + lambda_2 * nom1
            + lambda_3 * nom2
            - epsilon * x_bar.dot(x_bar)
        )
        # neg saturation
        prog.AddSosConstraint(
            -Vdot_minus
            + lambda_a * (V - rho)
            + lambda_1 * neg
            - epsilon * x_bar.dot(x_bar)
        )
        # pos saturation
        prog.AddSosConstraint(
            -Vdot_plus
            + lambda_c * (V - rho)
            + lambda_4 * pos
            - epsilon * x_bar.dot(x_bar)
        )

    # Problem solution
    result = Solve(prog)
    # solver = CsdpSolver()
    # result = solver.Solve(prog)
    # print(prog) # usefull for debugging
    # print(result.get_solver_details())

    if verbose:
        env = {
            x_bar_1: goal[0],
            x_bar_2: goal[1],
            xd_bar_1: goal[2],
            xd_bar_2: goal[3],
        }

        print("-K(xBar): ")
        print(-K.dot(x_bar)[1].Evaluate(env))
        print("xBar: ")
        print(sym.Expression(x_bar[0]).Evaluate(env))
        print(sym.Expression(x_bar[1]).Evaluate(env))
        print(sym.Expression(x_bar[2]).Evaluate(env))
        print(sym.Expression(x_bar[3]).Evaluate(env))
        print("dotX (approximated) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(acc1_approx.Evaluate(env))
        print(acc2_approx.Evaluate(env))
        print("dotX (true) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(acc[0].Evaluate(env))
        print(acc[1].Evaluate(env))
        print("V")
        print(V.Evaluate(env))
        print("Vdot (approximated)")
        print(Vdot.Evaluate(env))
        print("Vdot check (approximated)")
        print(Vdot_check.Evaluate(env))
        print("Vdot true")
        print(Vdot_true.Evaluate(env))
        print("S")
        print(S)

    return result.GetSolution(rho), result.is_success()


def bisect_and_verify(
    model_par,
    goal,
    S,
    K,
    robot,
    hyper_params,
    rho_min=1e-10,
    rho_max=5,
    maxiter=15,
    verbose=False,
):
    """
    Simple bisection root finding for finding the RoA using the feasibility
    problem.
    The default values have been choosen after multiple trials.
    """

    for i in range(maxiter):
        # np.random.uniform(rho_min,rho_max)
        rho_probe = rho_min + (rho_max - rho_min) / 2
        _, res = estimate_roa_sos(
            model_par,
            goal,
            rho_probe,
            S,
            K,
            robot,
            taylor_deg=hyper_params["taylor_deg"],
            lambda_deg=hyper_params["lambda_deg"],
            mode=hyper_params["mode"],
        )
        if verbose:
            print("---")
            print("rho_min:   " + str(rho_min))
            print("rho_probe: " + str(rho_probe) + " verified: " + str(res))
            print("rho_max:   " + str(rho_max))
            print("---")
        if res:
            rho_min = rho_probe
        else:
            rho_max = rho_probe

    return rho_min


def estimate_roa_sos_constrained(
    model_par,
    goal,
    S,
    K,
    robot,
    taylor_deg=3,
    lambda_deg=2,
    verbose=False,
):
    """
    params      --> parameters of pendulum and controller
    taylor_deg  --> degree of the taylor approximation
    lamda_deg   --> degree of SOS lagrange multipliers

    It solves the equality constrained formulation of the SOS problem just for the completely unconstrained (no actuation limits) case.
    Surprisingly it usually just slightly overestimate the actual RoA.
    The computational time is the worst between the SOS-based method but it is still very convenient wrt the najafi one.
    On the other hand, a bad closed-loop dynamics makes the estimation to drammatically decrease.
    """

    active_motor_ind = None
    if robot == "acrobot":
        active_motor_ind = 1
    elif robot == "pendubot":
        active_motor_ind = 0

    plant = SymbolicDoublePendulumPydrake(model_pars=model_par)

    # Opt. Problem definition (Indeterminates in error coordinates)
    prog = MathematicalProgram()
    x_bar = prog.NewIndeterminates(4, "x_bar")
    x_bar_1 = x_bar[0]
    x_bar_2 = x_bar[1]
    xd_bar_1 = x_bar[2]
    xd_bar_2 = x_bar[3]

    rho = prog.NewContinuousVariables(1, "rho")[0]
    prog.AddCost(-rho)  # Aiming to maximize rho

    # Dynamics definition
    x = goal + x_bar  # state in physical coordinates
    u = -K.dot(x_bar)  # control input

    acc = plant.forward_dynamics(x, u)

    # Manipulator eqs (this is not in error coords)
    # q1 = x[0]
    # q2 = x[1]
    qd1 = x[2]
    qd2 = x[3]

    # Taylor approximation
    env = {x_bar_1: 0.0, x_bar_2: 0.0, xd_bar_1: 0.0, xd_bar_2: 0.0}
    qdd1_approx = sym.TaylorExpand(f=acc[0], a=env, order=taylor_deg)
    qdd2_approx = sym.TaylorExpand(f=acc[1], a=env, order=taylor_deg)

    f = np.array([[qd1], [qd2], [qdd1_approx], [qdd2_approx]])

    # Definition of the Lyapunov function and of its derivative
    V = x_bar.dot(S.dot(x_bar))
    Vdot = (V.Jacobian(x_bar).dot(f))[0]

    Vdot_check = 2 * np.dot(
        x_bar, np.dot(S, f.flatten())
    )  # Checking the effect of the approximation on Vdot
    f_true = np.array([qd1, qd2, acc[0], acc[1]])
    Vdot_true = 2 * np.dot(x_bar, np.dot(S, f_true.flatten()))

    # Multipliers definition
    lambda_b = prog.NewFreePolynomial(Variables(x_bar), lambda_deg).ToExpression()

    # Completely unconstrained dynamics
    prog.AddSosConstraint(((x_bar.T).dot(x_bar) ** 2) * (V - rho) + lambda_b * (Vdot))

    # Problem solution
    result = Solve(prog)
    # solver = CsdpSolver()
    # result = solver.Solve(prog)
    # print(prog) # usefull for debugging
    # print(result)

    if verbose:
        env = {
            x_bar_1: goal[0],
            x_bar_2: goal[1],
            xd_bar_1: goal[2],
            xd_bar_2: goal[3],
        }

        print("-K(xBar): ")
        print(-K.dot(x_bar)[1].Evaluate(env))
        print("xBar: ")
        print(sym.Expression(x_bar[0]).Evaluate(env))
        print(sym.Expression(x_bar[1]).Evaluate(env))
        print(sym.Expression(x_bar[2]).Evaluate(env))
        print(sym.Expression(x_bar[3]).Evaluate(env))
        print("dotX (approximated) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(qdd1_approx.Evaluate(env))
        print(qdd2_approx.Evaluate(env))
        print("dotX (true) ")
        print(qd1.Evaluate(env))
        print(qd2.Evaluate(env))
        print(acc[0].Evaluate(env))
        print(acc[1].Evaluate(env))
        print("V")
        print(V.Evaluate(env))
        print("Vdot (approximated)")
        print(Vdot.Evaluate(env))
        print("Vdot check (approximated)")
        print(Vdot_check.Evaluate(env))
        print("Vdot true")
        print(Vdot_true.Evaluate(env))
        print("S")
        print(S)

    return result.GetSolution(rho)


def calc_roa(
    model_par=model_parameters(),
    goal=[np.pi, 0, 0, 0],
    Q=np.diag((1.0, 1.0, 1.0, 1.0)),
    R=np.diag((1.0, 1.0)),
    roa_backend="najafi",
    najafi_evals=1000,
    robot="acrobot",
    save_dir="data/",
    plots=False,
    verbose=False,
):
    os.makedirs(save_dir)

    verification_hyper_params = {}
    if roa_backend == "sos":
        verification_hyper_params = {
            "taylor_deg": 3,
            "lambda_deg": 4,
            "mode": 0,
        }
    if roa_backend == "sos_con":
        verification_hyper_params = {
            "taylor_deg": 3,
            "lambda_deg": 2,
            "mode": 2,
        }
    if roa_backend == "sos_eq":
        verification_hyper_params = {
            "taylor_deg": 3,
            "lambda_deg": 3,
            "mode": 2,
        }
    controller = LQRController(model_pars=model_par)

    controller.set_cost_matrices(Q=Q, R=R)
    controller.init()
    K = np.array(controller.K)
    S = np.array(controller.S)

    rho_f = 0.0
    if roa_backend == "sos" or roa_backend == "sos_con":
        rho_f = bisect_and_verify(
            model_par,
            goal,
            S,
            K,
            robot,
            verification_hyper_params,
            verbose=verbose,
            rho_min=1e-10,
            rho_max=5,
            maxiter=15,
        )

    elif roa_backend == "sos_eq":
        rho_f = estimate_roa_sos_constrained(
            model_par,
            goal,
            S,
            K,
            robot,
            verification_hyper_params["taylor_deg"],
            verification_hyper_params["lambda_deg"],
            verbose=verbose,
        )

    elif roa_backend == "prob":
        plant = DoublePendulumPlant(model_pars=model_par)
        nan_check = lqr_check_isnotNaN(plant, controller)
        conf = {
            "x0Star": goal,
            "S": S,
            "xBar0Max": np.array([+0.5, +0.0, 0.0, 0.0]),
            "nSimulations": 250,
        }

        # create estimation object
        estimator = estimate_roa_probabilistic(conf, nan_check.sim_callback)

        # do the actual estimation
        rho_hist, simSuccesHist = estimator.doEstimate()
        rho_f = rho_hist[-1]

    elif roa_backend == "najafi":
        plant = DoublePendulumPlant(model_pars=model_par)
        rho_f = estimate_roa_najafi(plant, controller, goal, S, najafi_evals)
        # print(rho_f)

    vol = volEllipsoid(rho_f, S)

    np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
    np.savetxt(os.path.join(save_dir, "vol"), [vol])
    # np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

    np.savetxt(
        os.path.join(save_dir, "controller_par.csv"),
        [Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3], R[0, 0]],
    )

    par_dict = {
        "goal_pos1": goal[0],
        "goal_pos2": goal[1],
        "goal_vel1": goal[2],
        "goal_vel2": goal[3],
        "Q1": float(Q[0][0]),
        "Q2": float(Q[1][1]),
        "Q3": float(Q[2][2]),
        "Q4": float(Q[3][3]),
        "R": float(R[0][0]),
        "roa_backend": roa_backend,
        "najafi_evaluations": najafi_evals,
    }

    model_par.save_dict(os.path.join(save_dir, "model_pars.yml"))

    with open(os.path.join(save_dir, "roa_parameters.yml"), "w") as f:
        yaml.dump(par_dict, f)

    if verbose:
        print("The estimated RoA volume is: " + str(vol))
        print("The major ellisoid axis is: " + str(rho_f))

    if plots:
        plotEllipse(
            goal[0],
            goal[1],
            0,
            1,
            rho_f,
            S,
            save_to=os.path.join(save_dir, "roaplot"),
            show=False,
        )
    return vol
