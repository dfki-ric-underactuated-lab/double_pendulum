import os
from pathlib import Path

import numpy as np

import pydrake.math as pm
from pydrake.solvers import MathematicalProgram, SnoptSolver
from pydrake.autodiffutils import AutoDiffXd


from double_pendulum.trajectory_optimization.direct_collocation import dircol_utils
from double_pendulum.utils.urdfs import generate_urdf

from scipy.interpolate import CubicHermiteSpline, interp1d

class DirCol():
    """This class formulates and solves the mathematical program for trajectory optimization
    via direct collocation. It uses drake for parsing the plant from a urdf as well as formulating
    and solving the optimization problem. The drake-internal direct collocation class is avoided for
    better control over the problem formulation. The class also implements resampling of the 
    found trajectory
    """
    def __init__(self,urdfPath,RobotType,modelPars,saveDir=".",nx=4,nu=2):
        """Class constructor

        Args:
            urdfPath (string): Path to the URDF-file
            RobotType (string): "acrobot", "pendubot" or "double_pendulum"
            modelPars (model_parameters()): parameters from system identification
            saveDir (str, optional): Path to save results into. Defaults to ".".
            nx (int, optional): Length of state vektor. Defaults to 4.
            nu (int, optional): Length of control input vector. Defaults to 2.
        """
        self.urdf_path = os.path.join(saveDir, RobotType + ".urdf")
        generate_urdf(urdfPath, self.urdf_path, model_pars=modelPars)
        self.RobotType = RobotType

        meshes_path = os.path.join(Path(urdfPath).parent, "meshes")
        os.system(f"cp -r {meshes_path} {saveDir}")

        self.plant, self.context, self. scene_graph = dircol_utils.create_plant_from_urdf(urdfPath)
        self.plant_ad = self.plant.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()
        self.prog = MathematicalProgram()
        self.nx = nx
        self.nu = nu

        if self.RobotType == "acrobot":
            self.B = np.array([[0., 1.], [0., 0.]])
        elif self.RobotType == "pendubot": 
            self.B = np.array([[1., 0.], [0., 0.]])
        else:
            self.B = np.array([[1., 0.], [0., 1.]])

    def EquationOfMotion(self,x,u,plant,context):
        """Computes the plant dynamics based on current state and input. Uses drake to extract matrices

        Args:
            x (array): current state. x.shape=(nx,)
            u (array): current input. u.shape=(nu,)
            plant (pydrake.plant): plant
            context (pydrake.context): plant context

        Returns:
            array: system dynamics xdot=(qd,qdd)
        """
        q = x[0:2]
        qd = x[2:4]

        plant.SetPositions(context,q)
        plant.SetVelocities(context,qd)

        M = plant.CalcMassMatrixViaInverseDynamics(context)
        Cv = plant.CalcBiasTerm(context)
        tauG = plant.CalcGravityGeneralizedForces(context)

        qdd = np.dot(pm.inv(M),tauG + self.B.dot(u) - Cv)
        
        return np.concatenate((qd,qdd))


    def CollocationConstraint(self,vars):
        """Helper function to compute collocation constraint. Adds support for 
        calls of plant within constraints
        """
        assert vars.size == 3 * self.nx + 3*self.nu + 1
        split_at = [
            self.nx,
            2 * self.nx,
            3 * self.nx,
            3 * self.nx + self.nu,
            3 * self.nx + 2 * self.nu,
            3 * self.nx + 3 * self.nu,
        ]
        xk,xk1,xk_half,uk,uk1,uk_half,h = np.split(vars, split_at)
        
        plant = (
            self.plant_ad if isinstance(vars[0], AutoDiffXd) else self.plant
        )
        context = (
            self.context_ad if isinstance(vars[0], AutoDiffXd) else self.context
        )

        return xk1 - xk - h/6 * (self.EquationOfMotion(xk,uk,plant,context) + 4 * self.EquationOfMotion(xk_half,uk_half,plant,context) + self.EquationOfMotion(xk1,uk1,plant,context))
    
    def InterpolationConstraint(self,vars):
        """Helper function to compute interpolation constraint. Adds support for 
        calls of plant within constraints
        """
        assert vars.size == 3 * self.nx + 2 * self.nu + 1
        split_at = [
            self.nx,
            2 * self.nx,
            3 * self.nx,
            3 * self.nx + self.nu,
            3 * self.nx + 2 * self.nu,
        ]
        xk,xk1,xk_half,uk,uk1,h = np.split(vars,split_at)

        plant = (
            self.plant_ad if isinstance(vars[0], AutoDiffXd) else self.plant
        )
        context = (
            self.context_ad if isinstance(vars[0], AutoDiffXd) else self.context
        )

        return 0.5 * (xk + xk1) + h/8 * (self.EquationOfMotion(xk,uk,plant,context) - self.EquationOfMotion(xk1,uk1,plant,context)) - xk_half


    def MathematicalProgram(self,N,Q,R,wh,h_min,h_max,x0,xf,torque_limit,X_initial,U_initial,h_initial):
        """Constructs the mathematical program and solves it using SNOPT from drake.
        Problem formulation done as described in paper by Matthew Kelly(2017), DOI:10.1137/16M1062569
        Stores solutions as class variables

        Args:
            N (int): Number of knot points
            Q (array): weighting matrix of state
            R (_type_): weighting matrix of input
            wh (_type_): weighting factor of time step length
            h_min (_type_): lower bound time step length
            h_max (_type_): upper bound time step length
            x0 (_type_): initial state
            xf (_type_): final state
            torque_limit (_type_): lower and upper bound input torque [Nm]
            X_initial (_type_): initial guess state trajectory
            U_initial (_type_): initial guess control trajectory
            h_initial (_type_): initial guess time step length
        """
        self.N = N
        X = self.prog.NewContinuousVariables(self.nx,N,"X") # knot points state
        U = self.prog.NewContinuousVariables(self.nu,N,"U") # knot points control input
        X_half = self.prog.NewContinuousVariables(self.nx,N-1,"X_half") # collocation points state
        U_half = self.prog.NewContinuousVariables(self.nu,N-1,"U_half") # collocation points control inputs
        h = self.prog.NewContinuousVariables(1,"h") # distances between knot points 
        for i in range(0,N-1):
            self.prog.AddCost(np.dot((X[:,i]-xf).T,Q.dot(X[:,i]-xf)) + np.dot(U[:,i].T,R.dot(U[:,i]))+wh*h[0]**2)
            # collocation constraint
            vars_coll = np.concatenate((
                X[:,i],
                X[:,i+1],
                X_half[:,i],
                U[:,i],
                U[:,i+1],
                U_half[:,i],
                np.array(h[0],ndmin=1) #force h to be 1d
            ))
            self.prog.AddConstraint(
                self.CollocationConstraint, lb=[0] * self.nx, ub = [0]*self.nx, vars = vars_coll
            )
            # interpolation constraint
            vars_interpol = np.concatenate((
                X[:,i],
                X[:,i+1],
                X_half[:,i],
                U[:,i],
                U[:,i+1],
                np.array(h[0],ndmin=1) #force h to be 1d
            ))
            self.prog.AddConstraint(
                self.InterpolationConstraint, lb = [0] * self.nx, ub = [0] * self.nx, vars = vars_interpol
            )
            #torque limits
            self.prog.AddBoundingBoxConstraint(-torque_limit, torque_limit, U[:,i]) #not for all states here?

        #time step limits
        self.prog.AddBoundingBoxConstraint(h_min, h_max, h)
        #initial and final conditions
        self.prog.AddBoundingBoxConstraint(x0,x0,X[:,0])
        self.prog.AddBoundingBoxConstraint(xf,xf,X[:,-1])
        self.prog.AddBoundingBoxConstraint(np.zeros(2),np.zeros(2),U[:,-1])
        #initial guess
        self.prog.SetInitialGuess(X, X_initial)
        self.prog.SetInitialGuess(X_half, np.zeros_like(X_half))
        self.prog.SetInitialGuess(U,U_initial)
        self.prog.SetInitialGuess(U_half, np.zeros_like(U_half))
        self.prog.SetInitialGuess(h[0], h_initial)
        #solve programm
        solver = SnoptSolver()
        self.result = solver.Solve(self.prog)
        print(f"Solution found? {self.result.is_success()}.")
        #save solutions
        self.X_sol = self.result.GetSolution(X)
        self.X_half_sol = self.result.GetSolution(X_half)
        self.U_sol = self.result.GetSolution(U)
        self.U_half_sol = self.result.GetSolution(U_half)
        self.h_sol = self.result.GetSolution(h)
    

    def ComputeTrajectory(self,freq):
        """Resamples trajectories using scipy CubicHermiteSpline for state trajectory and 1st order hold
        for input trajectory

        Args:
            freq (float64): sampling frequency

        Returns:
            t (array): time vector
            x (array): resampled state trajectory
            u (array): resampled input trajectory
        """
        T = np.array([i*self.h_sol[0] for i in range(self.N)])
        f = np.array([self.EquationOfMotion(x=self.X_sol[:,i],u=self.U_sol[:,i],plant=self.plant,context=self.context) for i in range(self.N)])
        x_interp = CubicHermiteSpline(x=T, y=self.X_sol.T,dydx=f)
        u_shoulder_interp = interp1d(x=T,y=self.U_sol.T[:,0])
        u_ellbow_interp = interp1d(x=T,y=self.U_sol.T[:,1])
        
        n = int(T[-1] * freq)
        t = np.linspace(start=0,stop=T[-1],num=n)
        x = x_interp(t)
        u = np.column_stack((u_shoulder_interp(t),u_ellbow_interp(t)))

        return t,x,u
