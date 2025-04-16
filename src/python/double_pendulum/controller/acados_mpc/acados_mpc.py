import os
import tempfile
from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver
import casadi as cas
import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.acados_mpc.acados_model import PendulumModel
import yaml
import concurrent.futures
import sys
import numpy as np
import pandas as pd
from double_pendulum.controller.pid.point_pid_controller import PointPIDController

"""
File containts:

AcadosMpcController
integration of the acados ocp solevr into the dfki repo by using the abstract controller class

PendulumModel
acados model of the double pendulum
"""


class AcadosMpc(AbstractController):
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

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.coulomb_fric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            self.torque_limit = model_pars.tl

        self.pendulum_model = None
        self.warm_start = True


        # set default parameters
        self.set_start()
        self.set_goal()
        self.set_parameters()
        self.set_cost_parameters()
        self.set_velocity_constraints()

    def set_cost_parameters(
        self,
        Q_mat=2 * np.diag([0, 0, 0, 0]),
        Qf_mat=2 * np.diag([0, 0, 0, 0]),
        R_mat=2 * np.diag([0, 0]),
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

    def set_start(self, x0=[0, 0, 0, 0]):
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

    def set_goal(self, xf=[0, 0, 0, 0]):
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
        N_horizon=20,
        prediction_horizon=0.5,
        Nlp_max_iter=500,
        max_solve_time=1.0,
        solver_type="SQP_RTI",
        wrap_angle=False,
        warm_start=False,
        scaling=1,
        nonuniform_grid=False,
        use_energy_for_terminal_cost=False,
        fallback_on_solver_fail=False,
        cheating_on_inactive_joint=0,
        mpc_cycle_dt=0.01,
        pd_tracking=False,
        outer_cycle_dt=0.001,
        pd_KP=None,
        pd_KD=None,
        pd_KI=None,
        qp_solver_tolerance=None,
        qp_solver="PARTIAL_CONDENSING_HPIPM",
        hpipm_mode="SPEED",
        **kwargs,
    ):
        self.N_horizon = N_horizon
        self.prediction_horizon = prediction_horizon
        self.Nlp_max_iter = Nlp_max_iter
        self.max_solve_time = max_solve_time
        self.solver_type = solver_type
        self.wrap_angle = wrap_angle
        self.warm_start = warm_start
        self.scaling = (
            np.full(N_horizon, scaling) if type(scaling).__name__ == "int" else scaling
        )
        self.nonuniform_grid = nonuniform_grid
        self.use_energy_for_terminal_cost = use_energy_for_terminal_cost
        self.fallback_on_solver_fail = fallback_on_solver_fail
        self.cheating_on_inactive_joint = cheating_on_inactive_joint
        self.hpipm_mode = hpipm_mode
        self.qp_solver_tolerance = qp_solver_tolerance
        self.qp_solver = qp_solver
        self.options = kwargs

        if pd_tracking:
            self.pd_tracking_controller = PointPIDController(
                self.torque_limit, outer_cycle_dt
            )
            self.pd_tracking_controller.set_parameters(pd_KP, pd_KI, pd_KD)
        else:
            self.pd_tracking_controller = None

        self.mpc_cycle_dt = mpc_cycle_dt

        self.async_mpc_future = None
        self.spinner = concurrent.futures.ThreadPoolExecutor(1)
        #print(f"mpc_cycle_dt={mpc_cycle_dt}, outer_cycle_dt={outer_cycle_dt}")

    def init_(self):
        self.pendulum_model = PendulumModel(
            mass=self.mass,
            length=self.length,
            damping=self.damping,
            coulomb_fric=self.coulomb_fric,
            inertia=self.inertia,
            center_of_mass=self.com,
            actuated_joint=(
                -1 if self.cheating_on_inactive_joint else np.argmax(self.torque_limit)
            ),
        )
        self.setup_solver()

        if self.warm_start:
            self.initial_iter()

        if self.pd_tracking_controller:
            self.pd_tracking_controller.init_()

        self.last_mpc_run_t = -float("inf")
        self.mpc_x = np.zeros(4)
        self.mpc_u = np.zeros(2)
        self.solve_times = []
        self.solve_times_T = []
        self.solve_times_rti_prepare = []
        self.solve_times_rti_feedback = []
        

    def setup_solver(self):
        DDP = self.solver_type == "DDP"
        self.use_RTI = self.solver_type == "SQP_RTI"

        ocp = AcadosOcp()
        ocp.model = self.pendulum_model.acados_model()

        nx = ocp.model.x.rows()
        nu = ocp.model.u.rows()

        ocp.cost.W_e = self.Qf_mat
        ocp.cost.yref_e = self.xf
        ocp.cost.yref = np.hstack([self.xf, np.zeros((nu,))])
        ocp.cost.W = cas.diagcat(self.Q_mat, self.R_mat).full()

        if self.wrap_angle:
            Endeffektor_desired = np.zeros(2)
            Endeffektor_pos = cas.SX.sym("endeff_pos", 2)

            Endeffektor_pos[0] = cas.cos(ocp.model.x[0]) * self.length[0] + cas.cos(
                ocp.model.x[1] * self.length[1]
            )
            Endeffektor_pos[1] = cas.sin(ocp.model.x[0]) * self.length[0] + cas.sin(
                ocp.model.x[1] * self.length[1]
            )

            Endeffektor_desired[0] = np.cos(self.xf[0]) * self.length[0] + np.cos(
                self.xf[1] * self.length[1]
            )
            Endeffektor_desired[1] = np.sin(self.xf[0]) * self.length[0] + np.sin(
                self.xf[0] * self.length[1]
            )

            ocp.cost.cost_type = "NONLINEAR_LS"
            ocp.model.cost_y_expr = cas.vertcat(
                Endeffektor_pos, ocp.model.x[2], ocp.model.x[3], ocp.model.u
            )
            ocp.cost.cost_type_e = "NONLINEAR_LS"
            ocp.model.cost_y_expr_e = cas.vertcat(
                Endeffektor_pos, ocp.model.x[2], ocp.model.x[3]
            )
            ocp.cost.yref_e = np.hstack([Endeffektor_desired, self.xf[2], self.xf[3]])
            ocp.cost.yref = np.hstack(
                [Endeffektor_desired, self.xf[2], self.xf[3], np.zeros((nu,))]
            )

        elif DDP:
            ocp.cost.cost_type = "NONLINEAR_LS"
            ocp.model.cost_y_expr = cas.vertcat(ocp.model.x, ocp.model.u)
            ocp.cost.cost_type_e = "NONLINEAR_LS"
            ocp.model.cost_y_expr_e = cas.vertcat(ocp.model.x)
        else:
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.Vx = np.zeros((nx + nu, nx))
            ocp.cost.Vx[:nx, :] = np.eye(nx)
            ocp.cost.Vu = np.zeros((nx + nu, nu))
            ocp.cost.Vu[nx:, :] = np.eye(nu)
            ocp.cost.cost_type_e = "LINEAR_LS"
            ocp.cost.Vx_e = np.eye(nx)

        if self.use_energy_for_terminal_cost:
            ocp.cost.cost_type_e = "NONLINEAR_LS"
            ocp.model.cost_y_expr_e = cas.vertcat(
                ocp.model.K1, ocp.model.K2, ocp.model.P1, ocp.model.P2
            )

            k1, k2 = self.pendulum_model.kinetic_energy(self.xf)
            p1, p2 = self.pendulum_model.potential_energy(self.xf)

            ocp.cost.yref_e = np.array([k1, k2, p1, p2]).reshape((4, 1))

        ocp.constraints.lbu = -np.array(self.torque_limit)
        ocp.constraints.ubu = np.array(self.torque_limit)

        if self.v_max:
            ocp.constraints.ubx = np.full(2, self.v_max)
            ocp.constraints.lbx = -np.full(2, self.v_max)
            ocp.constraints.idxbx = np.array([2, 3])

        if self.v_final:
            ocp.constraints.lbx_e = -np.full(2, self.v_final)
            ocp.constraints.ubx_e = np.full(2, self.v_final)
            ocp.constraints.idxbx_e = np.array([2, 3])

        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.x0 = self.x0

        # solver options
        ocp.solver_options.qp_solver = self.qp_solver
        if self.qp_solver_tolerance:
            ocp.solver_options.qp_solver_tol_comp = self.qp_solver_tolerance
            ocp.solver_options.qp_solver_tol_eq = self.qp_solver_tolerance
            ocp.solver_options.qp_solver_tol_ineq = self.qp_solver_tolerance
            ocp.solver_options.qp_solver_tol_stat = self.qp_solver_tolerance
        if not DDP:
            ocp.solver_options.qp_solver_cond_N = int(self.N_horizon / 2)
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.hpipm_mode = self.hpipm_mode
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.nlp_solver_type = self.solver_type
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.regularize_method
        ocp.solver_options.nlp_solver_max_iter = self.Nlp_max_iter
        ocp.solver_options.with_adaptive_levenberg_marquardt = True
        ocp.solver_options.N_horizon = self.N_horizon
        ocp.solver_options.tf = self.prediction_horizon
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.nlp_solver_warm_start_first_qp = False
        ocp.solver_options.print_level = 0
        ocp.solver_options.timeout_max_time = self.max_solve_time
        ocp.solver_options.as_rti_iter = 1
        ocp.parameter_values = np.array([100])
        # ocp.p_global_values = np.array([100])
        self.last_u = np.zeros([self.N_horizon, 2])
        self.time_grid = np.linspace(0, self.prediction_horizon, self.N_horizon)

        if self.nonuniform_grid:
            time_steps = np.linspace(0, 1, self.N_horizon + 1)[1:]
            time_steps = self.prediction_horizon * time_steps / sum(time_steps)
            ocp.solver_options.time_steps = time_steps
            self.time_grid = time_steps

        #print(f"time grid is {self.time_grid}")
        self.last_good_solution_time = 0.0

        for key, value in self.options.items():
            try:
                getattr(ocp.solver_options, key)
                setattr(ocp.solver_options, key, value)
            except AttributeError:
                pass

        if DDP:
            ocp.solver_options.nlp_solver_type = "DDP"
            ocp.translate_to_feasibility_problem(
                keep_x0=True, keep_cost=True, parametric_x0=False
            )

        temp_dir = tempfile.TemporaryDirectory()
        filename = temp_dir.name + "/acados_ocp.json"
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=filename)

        #for i in range(self.N_horizon):
        #    acados_ocp_solver.cost_set(i, "scaling", int(self.scaling[i]))
        #    # ocp_solver.set(i, "p_global", 100)
        #    acados_ocp_solver.set(i, "p", 100)

        self.ocp_solver = acados_ocp_solver

        if self.use_RTI:
            self.ocp_solver.options_set("rti_phase", 1)
            self.async_mpc_future = self.spinner.submit(self.ocp_solver.solve())

    def initial_iter(self):
        # do some initial iterations to start with a good initial guess
        num_iter_initial = 30
        for _ in range(num_iter_initial):
            self.ocp_solver.solve_for_x0(x0_bar=self.x0, fail_on_nonzero_status=False, print_stats_on_failure=False)

    def reset_(self):
        self.ocp_solver.reset()

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
        if not self.wrap_angle: # wrap the current state so the controller does not need to cricle back
            x[0] = (x[0] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
            x[1] = (x[1] + 2 * np.pi) % (4 * np.pi) - 2 * np.pi
            x[2] = np.clip(x[2], -25, 25)
            x[3] = np.clip(x[3], -25, 25)

        if (t - self.last_mpc_run_t) > self.mpc_cycle_dt:  # inner loop
            # check if last mpc call did converge within time
            # if self.async_mpc_future and self.async_mpc_future.done():
            #        self.mpc_u = self.ocp_solver.get(0, "u")
            #        self.mpc_x = self.ocp_solver.get(1, "x")
            # else:
            #    print(f"MPC did not return in time! {t} with {t - self.last_mpc_run_t}")
            # spin next cycle:
            # self.async_mpc_future = self.spinner.submit(self.ocp_solver.solve_for_x0(x0_bar = x, fail_on_nonzero_status=False))
            self.ocp_solver.set(0, "lbx", x)
            self.ocp_solver.set(0, "ubx", x)

            if self.use_RTI:
                if not self.async_mpc_future.done():
                    result = self.async_mpc_future.result()
                self.ocp_solver.options_set("rti_phase", 2)
                self.ocp_solver.solve()
                self.solve_times.append(self.ocp_solver.get_stats('time_tot'))
                self.solve_times_rti_feedback.append(self.ocp_solver.get_stats('time_feedback'))
                self.solve_times_rti_prepare.append(self.ocp_solver.get_stats('time_preparation'))
                self.solve_times_T.append(t)
                self.mpc_u = self.get_x0(x, t)
                self.ocp_solver.options_set("rti_phase", 1)
                self.async_mpc_future = self.spinner.submit(lambda : self.ocp_solver.solve())
            else:
                self.ocp_solver.solve()
                self.solve_times.append(self.ocp_solver.get_stats('time_tot'))
                self.solve_times_T.append(t)
                self.mpc_u = self.get_x0(x, t)
                # self.mpc_u = self.ocp_solver.solve_for_x0(x0_bar = x, fail_on_nonzero_status=False)

            self.last_mpc_run_t = t
            self.mpc_x = self.ocp_solver.get(1, "x")
        # outer loop
        if self.pd_tracking_controller:
            self.pd_tracking_controller.set_goal(self.mpc_x)
            return self.pd_tracking_controller.get_control_output(x, t)
        else:
            return self.mpc_u

    def get_x0(self, x, t):
        status = self.ocp_solver.get_status()
        if self.fallback_on_solver_fail:
            if status == 4 or status == 1 or status == 6:
                dt = 0
                for i in range(len(self.time_grid)):
                    dt += self.time_grid[i]
                    if (t - self.last_good_solution_time) <= dt:
                        return self.last_u[i]

                return np.zeros(2)
            else:
                for i in range(self.N_horizon):
                    self.last_u[i, :] = self.ocp_solver.get(i, "u")
                self.last_good_solution_time = t

            return self.last_u[0]
        else:
            return self.ocp_solver.get(0, "u")

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
            "scaling": self.scaling.tolist(),
            "max_solve_time": self.max_solve_time,
            "solver_type": self.solver_type,
            "wrap_angle": self.wrap_angle,
            "warm_start": self.warm_start,
            "fallback_on_solver_fail": self.fallback_on_solver_fail,
            "nonuniform_grid": self.nonuniform_grid,
            "cheating_on_inactive_joint": self.cheating_on_inactive_joint,
            "mpc_cycle_dt": self.mpc_cycle_dt,
            "qp_solver_tolerance": self.qp_solver_tolerance,
            "qp_solver": self.qp_solver,
            "hpipm_mode": self.hpipm_mode,
        }

        model_pars.update(self.options)

        with open(
            os.path.join(save_dir, "controller_ilqr_mpc_parameters.yml"), "w"
        ) as f:
            yaml.dump(model_pars, f)

        # self.ocp_solver.acados_ocp.dump_to_json(os.path.join(save_dir, "acados_ocp_descr.json"))

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
        x = np.zeros((self.N_horizon + 1, nx))
        u = np.zeros((self.N_horizon, nu))

        x[0, :] = self.ocp_solver.get(0, "x")
        for i in range(self.N_horizon):
            t[i] = i * self.prediction_horizon / self.N_horizon
            u[i, :] = self.ocp_solver.get(i, "u")
            x[i + 1, :] = self.ocp_solver.get(i, "x")

        return t, x, u

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

        init_traj = pd.read_csv(
            "/home/blanka/acados/examples/acados_python/own/trajectory.csv"
        )
        t = init_traj["time"].to_numpy()
        X_init = init_traj[["pos1", "pos2", "vel1", "vel2"]].to_numpy()
        U_init = init_traj[["tau1", "tau2"]].to_numpy()

        for i in range(1, self.N_horizon):
            self.ocp_solver.set(i, "x", X_init[i - 1, :])
            self.ocp_solver.set(i, "u", U_init[i, :])

        self.warm_start = False
