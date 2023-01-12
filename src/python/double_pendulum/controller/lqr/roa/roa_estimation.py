import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram#, Solve
from pydrake.solvers.csdp import CsdpSolver
from pydrake.symbolic import Variables
import pydrake.symbolic as sym

from double_pendulum.controller.lqr.roa.ellipsoid import quadForm, sampleFromEllipsoid


class probTIROA:
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
            x0 = self.x0Star+x0Bar  # error to absolute coords

            simSuccess = self.simClbk(x0)

            if not simSuccess:
                self.rhoHist.append(JStar0)
            else:
                self.rhoHist.append(self.rhoHist[-1])

            self.simSuccessHist.append(simSuccess)

        return self.rhoHist, self.simSuccessHist


def SosDoublePendulumDynamics(params,x, u, u_minus_vec, u_plus_vec, lib, robot = "acrobot"):

    """
    Facility in order to deal with the Dynamics definition in the SOS estimation method.
    """

    ## Model parameters
    I1  = params["I"][0]
    I2  = params["I"][1]
    m1  = params["m"][0]
    m2  = params["m"][1]
    l1  = params["l"][0]
    # l2  = params["l"][1] # because use of r2
    r1  = params["lc"][0]
    r2  = params["lc"][1]
    b1  = params["b"][0]
    b2  = params["b"][1]
    fc1 = params["fc"][0]
    fc2 = params["fc"][1]
    g   = params["g"]

    q    = x[0:2] #Change of coordinates for manipulator eqs (this is not in error coords)
    qd   = x[2:4]
    q1   = q[0]
    q2   = q[1]
    qd1  = qd[0]
    qd2  = qd[1]

    m11 = I1 + I2 + m2*l1**2 + 2*m2*l1*r2*lib.cos(q2) # mass matrix
    m12 = I2 + m2 *l1 * r2 * lib.cos(q2)
    m21 = I2 + m2 *l1 * r2 * lib.cos(q2)
    m22 = I2
    # M   = np.array([[m11,m12],
    #                 [m21,m22]])
    det_M = m22*m11-m12*m21
    M_inv = (1/det_M) * np.array([  [m22,-m12],
                                    [-m21,m11]])

    c11 =  -2 * m2 * l1 * r2 * lib.sin(q2) * qd2 # coriolis matrix
    c12 = -m2 * l1 * r2 * lib.sin(q2) * qd2
    c21 =  m2 * l1 * r2 * lib.sin(q2) * qd1
    c22 = 0
    C   = np.array([[c11,c12],
                    [c21,c22]])

    sin12 = (lib.sin(q1)*lib.cos(q2)) + (lib.sin(q2)*lib.cos(q1)) # sen(q1+q2) = sen(q1)cos(q2) + sen(q2)cos(q1)
    g1 = -m1*g*r1*lib.sin(q1) - m2*g*(l1*lib.sin(q1) + r2*sin12) # gravity matrix
    g2 = -m2*g*r2*sin12
    G  = np.array([g1,g2])


    if lib == sym:
        f1 = b1*qd1 + fc1*lib.atan(100*qd1) # coloumb vector symbolic for taylor
        f2 = b2*qd2 + fc2*lib.atan(100*qd2)
        F = np.array([f1,f2])
    elif lib == np:
        f1 = b1*qd1 + fc1*lib.arctan(100*qd1) # coloumb vector nominal
        f2 = b2*qd2 + fc2*lib.arctan(100*qd2)
        F = np.array([f1,f2])

    if robot == "acrobot":
        B  = np.array([[0,0],[0,1]]) # b matrix acrobot
    elif robot == "pendubot":
        B  = np.array([[1,0],[0,0]]) # b matrix pendubot

    f_exp_acc       = M_inv.dot(   B.dot(u) + G - C.dot(qd) - F ) # nominal and saturated explicit dynamics
    f_exp_acc_minus = M_inv.dot(   B.dot( u_minus_vec   ) + G - C.dot(qd) - F )
    f_exp_acc_plus  = M_inv.dot(   B.dot( u_plus_vec    ) + G - C.dot(qd) - F )

    return f_exp_acc, f_exp_acc_minus, f_exp_acc_plus

def verify_double_pendulum_rho(rho, params, S, K, robot, taylor_deg=3,
                               lambda_deg=4, mode=2, verbose=False,
                               x_bar_eval=[np.pi, 0, 0, 0]):
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
    S = S  # params["S"]
    K = K  # params["K"]

    # Saturation parameters
    u_plus_vec = np.array(params["tau_max"])
    u_minus_vec = - np.array(params["tau_max"])

    # Opt. Problem definition (Indeterminates in error coordinates)
    prog = MathematicalProgram()
    x_bar = prog.NewIndeterminates(4, "x_bar")
    x_bar_1 = x_bar[0]
    x_bar_2 = x_bar[1]
    xd_bar_1 = x_bar[2]
    xd_bar_2 = x_bar[3]

    # Dynamics definition
    x_star      = np.array([np.pi,0,0,0]) # desired state in physical coordinates
    x           = x_star+x_bar # state in physical coordinates
    u = -K.dot(x_bar) # control input

    f_exp_acc,f_exp_acc_minus,f_exp_acc_plus = SosDoublePendulumDynamics(params,x, u, u_minus_vec, u_plus_vec, sym, robot)
    q = x[0:2] # Manipulator eqs (this is not in error coords)
    qd = x[2:4]
    q1 = q[0]
    q2 = q[1]
    qd1 = qd[0]
    qd2 = qd[1]

    # Taylor approximation
    env = {x_bar_1: 0,
           x_bar_2: 0,
           xd_bar_1: 0,
           xd_bar_2: 0}
    qdd1_approx = sym.TaylorExpand(f=f_exp_acc[0], a=env, order=taylor_deg)
    qdd2_approx = sym.TaylorExpand(f=f_exp_acc[1], a=env, order=taylor_deg)
    if mode == 2:
        qdd1_approx_minus = sym.TaylorExpand(f=f_exp_acc_minus[0],
                                             a=env,
                                             order=taylor_deg)
        qdd2_approx_minus = sym.TaylorExpand(f=f_exp_acc_minus[1],
                                             a=env,
                                             order=taylor_deg)
        qdd1_approx_plus = sym.TaylorExpand(f=f_exp_acc_plus[0],
                                            a=env,
                                            order=taylor_deg)
        qdd2_approx_plus = sym.TaylorExpand(f=f_exp_acc_plus[1],
                                            a=env,
                                            order=taylor_deg)
    f = np.array([[qd1],
                  [qd2],
                  [qdd1_approx],
                  [qdd2_approx]])

    if mode == 2:
        f_minus = np.array([[qd1],
                            [qd2],
                            [qdd1_approx_minus],
                            [qdd2_approx_minus]])

        f_plus = np.array([[qd1],
                           [qd2],
                           [qdd1_approx_plus],
                           [qdd2_approx_plus]])

    # Definition of the Lyapunov function and of its derivative
    V = x_bar.dot(S.dot(x_bar))
    Vdot = (V.Jacobian(x_bar).dot(f))[0]
    if mode == 2:
        Vdot_minus = (V.Jacobian(x_bar).dot(f_minus))[0]
        Vdot_plus = (V.Jacobian(x_bar).dot(f_plus))[0]

    Vdot_check = 2*np.dot(x_bar, np.dot(S, f.flatten())) # Checking the effect of the approximation on Vdot
    f_true = np.array([qd1, qd2, f_exp_acc[0], f_exp_acc[1]])
    Vdot_true = 2*np.dot(x_bar, np.dot(S, f_true.flatten()))

    # Multipliers definition
    lambda_b = prog.NewSosPolynomial(Variables(x_bar), lambda_deg)[0].ToExpression()
    if mode == 2:
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
        prog.AddSosConstraint(-Vdot + lambda_b*(V-rho) - epsilon*x_bar.dot(x_bar))

    # Considering the input saturation in the constraints
    if mode == 2:
        if robot == "acrobot":
            nom1 = (+ K.dot(x_bar) + u_minus_vec)[1] # where both nom1 and nom2 are < 0, the nominal dynamics have to be fullfilled
            nom2 = (- K.dot(x_bar) - u_plus_vec)[1]
            neg = (- K.dot(x_bar) - u_minus_vec)[1]  # where this is < 0, the negative saturated dynamics have to be fullfilled
            pos = (+ K.dot(x_bar) + u_plus_vec)[1]   # where this is < 0, the positive saturated dynamics have to be fullfilled
        if robot == "pendubot":
            nom1 = (+ K.dot(x_bar) + u_minus_vec)[0] # where both nom1 and nom2 are < 0, the nominal dynamics have to be fullfilled
            nom2 = (- K.dot(x_bar) - u_plus_vec)[0]
            neg = (- K.dot(x_bar) - u_minus_vec)[0]  # where this is < 0, the negative saturated dynamics have to be fullfilled
            pos = (+ K.dot(x_bar) + u_plus_vec)[0]   # where this is < 0, the positive saturated dynamics have to be fullfilled

        prog.AddSosConstraint(-Vdot + lambda_b*(V-rho) + lambda_2*nom1 + lambda_3*nom2 - epsilon*x_bar.dot(x_bar))
        # neg saturation
        prog.AddSosConstraint(-Vdot_minus + lambda_a*(V - rho) + lambda_1*neg - epsilon*x_bar.dot(x_bar))
        # pos saturation
        prog.AddSosConstraint(-Vdot_plus + lambda_c*(V - rho) + lambda_4*pos - epsilon*x_bar.dot(x_bar))

    # Problem solution
    #result = Solve(prog)
    solver = CsdpSolver()
    result = solver.Solve(prog)
    # print(prog) # usefull for debugging
    # print(result.get_solver_details())

    if verbose:
        env = {x_bar_1: x_bar_eval[0],
               x_bar_2: x_bar_eval[1],
               xd_bar_1: x_bar_eval[2],
               xd_bar_2: x_bar_eval[3]}

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
        print(f_exp_acc[0].Evaluate(env))
        print(f_exp_acc[1].Evaluate(env))
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

    return result.is_success()


def bisect_and_verify(params, S, K, robot, hyper_params, rho_min=1e-10,
                      rho_max=5, maxiter=15, verbose=False):
    """
    Simple bisection root finding for finding the RoA using the feasibility
    problem.
    The default values have been choosen after multiple trials.
    """
    for i in range(maxiter):
        # np.random.uniform(rho_min,rho_max)
        rho_probe = rho_min+(rho_max-rho_min)/2
        res = verify_double_pendulum_rho(rho_probe,
                                         params,
                                         S,
                                         K,
                                         robot,
                                         taylor_deg=hyper_params["taylor_deg"],
                                         lambda_deg=hyper_params["lambda_deg"],
                                         mode=hyper_params["mode"])
        if verbose:
            print("---")
            print("rho_min:   "+str(rho_min))
            print("rho_probe: "+str(rho_probe)+" verified: "+str(res))
            print("rho_max:   "+str(rho_max))
            print("---")
        if res:
            rho_min = rho_probe
        else:
            rho_max = rho_probe

    return rho_min

def rho_equalityConstrained(params, S, K, robot, taylor_deg=3,
                            lambda_deg=2, verbose=False,
                            x_bar_eval=[np.pi, 0, 0, 0]):
    """
    params      --> parameters of pendulum and controller
    taylor_deg  --> degree of the taylor approximation
    lamda_deg   --> degree of SOS lagrange multipliers

    It solves the equality constrained formulation of the SOS problem just for the completely unconstrained (no actuation limits) case.
    Surprisingly it usually just slightly overestimate the actual RoA.
    The computational time is the worst between the SOS-based method but it is still very convenient wrt the najafi one.
    On the other hand, a bad closed-loop dynamics makes the estimation to drammatically decrease.
    """

    # LQR parameters
    S = S  # params["S"]
    K = K  # params["K"]

    # Saturation parameters
    u_plus_vec = np.array(params["tau_max"])
    u_minus_vec = - np.array(params["tau_max"])

    # Opt. Problem definition (Indeterminates in error coordinates)
    prog = MathematicalProgram()
    x_bar = prog.NewIndeterminates(4, "x_bar")
    x_bar_1 = x_bar[0]
    x_bar_2 = x_bar[1]
    xd_bar_1 = x_bar[2]
    xd_bar_2 = x_bar[3]

    rho = prog.NewContinuousVariables(1, "rho")[0]
    prog.AddCost(-rho) # Aiming to maximize rho

    # Dynamics definition
    x_star      = np.array([np.pi,0,0,0]) # desired state in physical coordinates
    x           = x_star+x_bar # state in physical coordinates
    u = -K.dot(x_bar) # control input

    f_exp_acc,f_exp_acc_minus,f_exp_acc_plus = SosDoublePendulumDynamics(params,x, u, u_minus_vec, u_plus_vec, sym, robot)
    q = x[0:2] # Manipulator eqs (this is not in error coords)
    qd = x[2:4]
    q1 = q[0]
    q2 = q[1]
    qd1 = qd[0]
    qd2 = qd[1]

    # Taylor approximation
    env = {x_bar_1: 0,
           x_bar_2: 0,
           xd_bar_1: 0,
           xd_bar_2: 0}
    qdd1_approx = sym.TaylorExpand(f=f_exp_acc[0], a=env, order=taylor_deg)
    qdd2_approx = sym.TaylorExpand(f=f_exp_acc[1], a=env, order=taylor_deg)

    f = np.array([[qd1],
                  [qd2],
                  [qdd1_approx],
                  [qdd2_approx]])

    # Definition of the Lyapunov function and of its derivative
    V = x_bar.dot(S.dot(x_bar))
    Vdot = (V.Jacobian(x_bar).dot(f))[0]

    Vdot_check = 2*np.dot(x_bar, np.dot(S, f.flatten())) # Checking the effect of the approximation on Vdot
    f_true = np.array([qd1, qd2, f_exp_acc[0], f_exp_acc[1]])
    Vdot_true = 2*np.dot(x_bar, np.dot(S, f_true.flatten()))

    # Multipliers definition
    lambda_b = prog.NewFreePolynomial(Variables(x_bar), lambda_deg).ToExpression()

    # Completely unconstrained dynamics
    prog.AddSosConstraint(((x_bar.T).dot(x_bar)**2)*(V - rho) + lambda_b*(Vdot))

    # Problem solution
    #result = Solve(prog)
    solver = CsdpSolver()
    result = solver.Solve(prog)
    # print(prog) # usefull for debugging
    # print(result)

    if verbose:
        env = {x_bar_1: x_bar_eval[0],
               x_bar_2: x_bar_eval[1],
               xd_bar_1: x_bar_eval[2],
               xd_bar_2: x_bar_eval[3]}

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
        print(f_exp_acc[0].Evaluate(env))
        print(f_exp_acc[1].Evaluate(env))
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
