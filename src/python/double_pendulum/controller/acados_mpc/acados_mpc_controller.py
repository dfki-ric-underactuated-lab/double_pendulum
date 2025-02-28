import os
from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver
import casadi as cas
import numpy as np
import pandas as pd
from double_pendulum.controller.abstract_controller import AbstractController
import yaml

"""
File containts:

AcadosMpcController
integration of the acados ocp solevr into the dfki repo by using the abstract controller class

PendulumModel
acados model of the double pendulum
"""
class AcadosMpcController(AbstractController): 
    def __init__(
        self,
        mass=[0.608, 0.630],
        length=[0.3, 0.2],
        com=[0.275, 0.166],
        damping=[0.081, 0.0],
        coulomb_fric=[0.093, 0.186],
        gravity=9.81,
        inertia=[0.05472, 0.02522],
        torque_limit=[0.0, 6.0],
        model_pars=None,
        cheating_on_inactive_joint=0,
    ):
        super().__init__()
        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.coulomb_fric = coulomb_fric
        self.gravity = gravity
        self.inertia = inertia
        self.torque_limit = torque_limit
        self.cheating_on_inactive_joint = cheating_on_inactive_joint

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.coulomb_fric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            self.torque_limit = model_pars.tl

        if self.torque_limit[0] > 0.0:
            self.active_act = 0
            self.Fmax = self.torque_limit[0]
        elif self.torque_limit[1] > 0.0:
            self.active_act = 1
            self.Fmax = self.torque_limit[1]
            
            
            
        self.pendulum_model = None 
        self.warm_start = True

        #set default parameters
        self.set_start()
        self.set_goal()
        self.set_parameters()
        self.set_cost_parameters()
        self.set_velocity_constraints()


    def set_cost_parameters(
        self,
        Q_mat = 2*np.diag([0,0, 0,0]),
        Qf_mat = 2*np.diag([0,0, 0,0]),
        R_mat = 2*np.diag([0, 0])
    ): 
        """
        Set cost matrices.

        Parameters
        ----------
        Q_mat : array_like, shape=(4,), dtype=float,
            cost matrices for angle1, angle2, velocity1, velocity2,
        Qf_mat : array_like, shape=(4,), dtype=float,
            final cost matrices for angle1, angle2, velocity1, velocity2,
        R_mat: array_like, shape=(2,), dtype=float,
            cost matrices for motor_torque1, motor_torque2
        """
        self.Q_mat = Q_mat
        self.Qf_mat = Qf_mat 
        self.R_mat = R_mat
    
    def set_velocity_constraints(self, v_max=None, v_final=None):
        """
        Set constraints on velocity matrices.

        Parameters
        ----------
        v_max : float,
            maximum velocity of the double pendulum
        v_final : array_like, shape=(2,), dtype=float,
             final velocity of the double pendulum for velocity1, velocity2
        """
        self.v_max = v_max
        self.v_final = v_final
        
    def set_start(self, x0=[0,0,0,0]):
        """
        Set start state

        Parameters
        ----------
        x0 : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.x0 = np.array(x0)
        
    def set_goal(self, xf=[0,0,0,0]):
        """
        Set desired state

        Parameters
        ----------
        xf : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.xf = np.array(xf)
        
        
    def set_parameters(
        self,
        N_horizon=1000,
        prediction_horizon=4.0,
        Nlp_max_iter=500,
        solver_type = "SQP",
        scaling = 1,
        wrap_angle=False,
        **kwargs
    ):
        self.N_horizon = N_horizon
        self.prediction_horizon = prediction_horizon
        self.Nlp_max_iter = Nlp_max_iter
        self.solver_type = solver_type
        self.wrap_angle = wrap_angle
        self.scaling = np.full(N_horizon, scaling) if type(scaling).__name__ == "int" else scaling 
        self.options = kwargs
        

    def init_(self):
        self.pendulum_model = PendulumModel(
            mass=self.mass,
            length=self.length, 
            damping=self.damping, 
            coulomb_fric=self.coulomb_fric,
            inertia=self.inertia,
            center_of_mass=self.com,
            actuated_joint= -1 if self.cheating_on_inactive_joint > 0 else self.active_act
        )
        
        self.statistics = {
            "t_min": 0,
            "t_max": 0,
            "t_med": 0
        }
        
        self.setup_solver()
        #if self.warm_start:
            #self.initial_iter()
        
    def setup_solver(self):
        DDP = self.solver_type == "DDP"
        
        ocp = AcadosOcp()
        ocp.model = self.pendulum_model.acados_model()
        
        nx = ocp.model.x.rows()
        nu = ocp.model.u.rows()
        
        if self.wrap_angle:
            x_wrapped = cas.SX.sym('xwrap', 4)
            x_wrapped[0] = cas.arctan2(cas.sin(ocp.model.x[0]), cas.cos(ocp.model.x[0]))
            x_wrapped[1] = cas.arctan2(cas.sin(ocp.model.x[1]), cas.cos(ocp.model.x[1]))
            x_wrapped[2] = ocp.model.x[2]
            x_wrapped[3] = ocp.model.x[3]
    
            ocp.cost.cost_type =  'NONLINEAR_LS'
            ocp.model.cost_y_expr = cas.vertcat(ocp.model.x, ocp.model.u)
            ocp.cost.cost_type_e =  'NONLINEAR_LS'
            ocp.model.cost_y_expr_e = cas.vertcat(ocp.model.x)
            
        elif DDP:
            ocp.cost.cost_type =  'NONLINEAR_LS'
            ocp.model.cost_y_expr = cas.vertcat(ocp.model.x, ocp.model.u)
            ocp.cost.cost_type_e =  'NONLINEAR_LS'
            ocp.model.cost_y_expr_e = cas.vertcat(ocp.model.x)
        else:
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.Vx = np.zeros((nx + nu,nx))
            ocp.cost.Vx[:nx, :] = np.eye(nx)
            ocp.cost.Vu = np.zeros((nx+nu, nu))
            ocp.cost.Vu[nx:, :] = np.eye(nu)
            ocp.cost.cost_type_e = 'LINEAR_LS'
            ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.W_e = self.Qf_mat
        ocp.cost.yref_e = self.xf
        ocp.cost.yref = np.hstack([self.xf, np.zeros((nu,))])
        ocp.cost.W = cas.diagcat(self.Q_mat, self.R_mat).full()
        
        if self.active_act == 0 :
            ocp.constraints.lbu = np.array([-self.Fmax, -self.cheating_on_inactive_joint])
            ocp.constraints.ubu = np.array([+self.Fmax, +self.cheating_on_inactive_joint])
        elif self.active_act == 1:
            ocp.constraints.lbu = np.array([-self.cheating_on_inactive_joint, -self.Fmax])
            ocp.constraints.ubu = np.array([+self.cheating_on_inactive_joint, +self.Fmax])
        elif self.active_act == -1:
            ocp.constraints.lbu = np.array([-self.Fmax,-self.Fmax])
            ocp.constraints.ubu = np.array([+self.Fmax,+self.Fmax])
            
        if self.v_max:
            ubv = np.full(2, self.v_max)
            ocp.constraints.ubx = ubv
            ocp.constraints.lbx = -ubv
            ocp.constraints.idxbx = np.array([2,3])
        
        if self.v_final:
            lbv = np.full(2, self.v_final)
            ubv = np.full(2, self.v_final)
            ocp.constraints.lbx_e = lbv
            ocp.constraints.ubx_e = ubv
            ocp.constraints.idxbx_e = np.array([2,3])
        
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.x0 = self.x0 
        
        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' 
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' 
        ocp.solver_options.nlp_solver_max_iter = self.Nlp_max_iter    
        ocp.solver_options.with_adaptive_levenberg_marquardt = True
        ocp.solver_options.N_horizon = self.N_horizon 
        ocp.solver_options.tf = self.prediction_horizon 
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.nlp_solver_warm_start_first_qp
        ocp.solver_options.print_level = 1
        
        for key, value in self.options.items():
            try:
                getattr(ocp.solver_options, key)
                setattr(ocp.solver_options, key, value)
            except  AttributeError:
                pass
        
        if DDP:
            ocp.solver_options.nlp_solver_type = 'DDP'
            ocp.translate_to_feasibility_problem(keep_x0=True, keep_cost=True, parametric_x0=False)
        
        ocp_solver = AcadosOcpSolver(ocp)
        acados_ocp_solver = AcadosOcpSolver(ocp)
        
        for i in range(self.N_horizon):
            ocp_solver.cost_set(i, "scaling", int(self.scaling[i]))
        
        self.ocp_solver = acados_ocp_solver

    def initial_iter(self):
        # do some initial iterations to start with a good initial guess
        num_iter_initial = 30
        for _ in range(num_iter_initial):
            self.ocp_solver.solve_for_x0(x0_bar = self.x0, fail_on_nonzero_status=False)

    def reset_(self):
        
        #reset stats
        self.statistics = {
            "t": np.array([]),
            "t_min": 0,
            "t_max": 0,
            "t_med": 0
        }
        
        # reset acados solver
        self.ocp_solver.reset()
        
    def update_Stats(self):
        print(self.statistics)
        self.statistics["t"].append(self.ocp_solver.get_stats('time_tot')*1000) #*1000 to millisec
        self.statistics["t_min"] =np.min(self.statistics["t"])
        self.statistics["t_max"] =np.max(self.statistics["t"])
        self.statistics["t_med"] =np.median(self.statistics["t"])
        #print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')
        
        
    def get_control_output_(self, x, t=None):
        """
        The function to compute the control input for the double pendulum's
        actuator(s).

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float, optional
            time, unit=[s]
            (Default value=None)

        Returns
        -------
        array_like
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        x = np.array(x)
        u0 =self.ocp_solver.solve_for_x0(x0_bar = x, fail_on_nonzero_status=False)
        #self.update_Stats()
        return u0

    def save_(self, save_dir):
        """
        Save controller parameters. Optional
        Can be overwritten by actual controller.

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        model_pars = {
            "N_horizon": self.N_horizon,
            "prediction_horizon": self.prediction_horizon,
            "Nlp_max_iter": self.Nlp_max_iter,
            "solver_type": self.solver_type,
            "scaling": self.scaling,
        }
        
        model_pars.update(self.options)
        
        with open(
            os.path.join(save_dir, "controller_ilqr_mpc_parameters.yml"), "w"
        ) as f:
            yaml.dump(model_pars, f)
            
        #self.ocp_solver.acados_ocp.dump_to_json(os.path.join(save_dir, "acados_ocp_descr.json"))


    def get_forecast(self):
        """
        Get a forecast trajectory as planned by the controller. Optional.
        Can be overwritten by actual controller.

        Returns
        -------
        list
            Time array
        list
            X array
        list
            U array
        """
        nx = self.ocp_solver.acados_ocp.dims.nx
        nu = self.ocp_solver.acados_ocp.dims.nu
    
        t = np.zeros(self.N_horizon)
        x = np.zeros((self.N_horizon+1, nx))
        u = np.zeros((self.N_horizon, nu))
        
        
        x[0, :] = self.ocp_solver.get(0, "x")
        for i in range(self.N_horizon):
            t[i] = i*self.prediction_horizon/self.N_horizon
            u[i,:] = self.ocp_solver.get(i, "u")
            x[i+1, :] = self.ocp_solver.get(i, "x")
            
        return t,x,u

    def get_init_trajectory(self):
        """
        Get an initial (reference) trajectory used by the controller. Optional.
        Can be overwritten by actual controller.

        Returns
        -------
        list
            Time array
        list
            X array
        list
            U array
        """
        return [], [], []
    
    def load_init_trajectory(self, csv_path):
        
        init_traj = pd.read_csv('/home/blanka/acados/examples/acados_python/own/trajectory.csv')
        t = init_traj["time"].to_numpy()
        X_init = init_traj[["pos1", "pos2", "vel1", "vel2"]].to_numpy()
        U_init = init_traj[["tau1", "tau2"]].to_numpy()

        for i in range(1, self.N_horizon):
            self.ocp_solver.set(i, "x", X_init[i-1,:])
            self.ocp_solver.set(i, "u", U_init[i,:])
            
        self.warm_start = False
        
    def print_stats(self):
        t = np.zeros(self.N_horizon)
        for i in range(self.N_horizon):
            t[i] = self.ocp_solver.get_stats('time_tot')

        t *= 1000
        print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')
  
        