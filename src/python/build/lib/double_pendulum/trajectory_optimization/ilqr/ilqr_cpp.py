import numpy as np

from cppilqr import cppilqr


class ilqr_calculator():
    """
    Class to calculate a trajectory for the acrobot or pendubot
    with iterative LQR. Implementation uses the python bindings of the
    C++ implementation for iLQR.

    Reference Paper for iLQR:
    https://www.scitepress.org/Link.aspx?doi=10.5220/0001143902220229
    """
    def _init__(self):

        # set default parameters
        self.set_model_parameters()
        self.set_parameters()
        self.set_cost_parameters()

    def set_model_parameters(self,
                             mass=[0.608, 0.630],
                             length=[0.3, 0.2],
                             com=[0.275, 0.166],
                             damping=[0.081, 0.0],
                             coulomb_fric=[0.093, 0.186],
                             gravity=9.81,
                             inertia=[0.05472, 0.02522],
                             torque_limit=[0.0, 6.0],
                             model_pars=None):
        """
        Set the model parameters of the robot.

        Parameters
        ----------
        mass : array_like, optional
            shape=(2,), dtype=float, default=[0.608, 0.630]
            masses of the double pendulum,
            [m1, m2], units=[kg]
        length : array_like, optional
            shape=(2,), dtype=float, default=[0.3, 0.2]
            link lengths of the double pendulum,
            [l1, l2], units=[m]
        com : array_like, optional
            shape=(2,), dtype=float, default=[0.275, 0.166]
            center of mass lengths of the double pendulum links
            [r1, r2], units=[m]
        damping : array_like, optional
            shape=(2,), dtype=float, default=[0.081, 0.0]
            damping coefficients of the double pendulum actuators
            [b1, b2], units=[kg*m/s]
        gravity : float, optional
            default=9.81
            gravity acceleration (pointing downwards),
            units=[m/s²]
        coulomb_fric : array_like, optional
            shape=(2,), dtype=float, default=[0.093, 0.186]
            coulomb friction coefficients for the double pendulum actuators
            [cf1, cf2], units=[Nm]
        inertia : array_like, optional
            shape=(2,), dtype=float, default=[0.05472, 0.02522]
            inertia of the double pendulum links
            [I1, I2], units=[kg*m²]
            if entry is None defaults to point mass m*l² inertia for the entry
        torque_limit : array_like, optional
            shape=(2,), dtype=float, default=[0.0, 6.0]
            torque limit of the motors
            [tl1, tl2], units=[Nm, Nm]
        model_pars : model_parameters object, optional
            object of the model_parameters class, default=None
            Can be used to set all model parameters above
            If provided, the model_pars parameters overwrite
            the other provided parameters
        """

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
            # self.Ir = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

    def set_parameters(self,
                       N=1000,
                       dt=0.005,
                       max_iter=100,
                       regu_init=100,
                       max_regu=10000.,
                       min_regu=0.01,
                       break_cost_redu=1e-6,
                       integrator="runge_kutta"):
        """set_parameters.
        Set parameters for the optimization

        Parameters
        ----------
        N : int
            number of timesteps for horizon
            (Default value = 1000)
        dt : float
            timestep for horizon
            (Default value = 0.005)
        max_iter : int
            maximum of optimization iterations per step
            (Default value = 1)
        regu_init : float
            inital regularization
            (Default value = 1.)
        max_regu : float
            maximum regularization
            (Default value = 10000.)
        min_regu : float
            minimum regularization
            (Default value = 0.01)
        break_cost_redu : float
            Stop the optimization at this cost reduction.
             (Default value = 1e-6)
        integrator : string
            string determining the integration method
            "euler" : Euler integrator
            "runge_kutta" : Runge Kutta integrator
             (Default value = "runge_kutta")
        """
        self.N = N
        self.dt = dt
        self.max_iter = max_iter
        self.regu_init = regu_init
        self.max_regu = max_regu
        self.min_regu = min_regu
        self.break_cost_redu = break_cost_redu

        if integrator == "euler":
            self.integrator_int = 0
        else:
            self.integrator_int = 1

    def set_cost_parameters(self,
                            sCu=[0.005, 0.005],
                            sCp=[0., 0.],
                            sCv=[0., 0.],
                            sCen=0.,
                            fCp=[1000., 1000.],
                            fCv=[10., 10.],
                            fCen=0.):
        """
        Set cost parameters used for optimization.

        Parameters
        ----------
        sCu : list
            shape=(2,), dtype=float
            stage cost weights for control input
            (Default value = [0.005, 0.005])
        sCp : list
            shape=(2,), dtype=float
            stage cost weights for position error
            (Default value = [0.,0.])
        sCv : list
            shape=(2,), dtype=float
            stage cost weights for velocity error
            (Default value = [0., 0.])
        sCen : float
            stage cost weight for energy error
            (Default value = 0.)
        fCp : list
            shape=(2,), dtype=float
            final cost weights for position error
             (Default value = [1000., 1000.])
        fCv : list
            shape=(2,), dtype=float
            final cost weights for velocity error
             (Default value = [10., 10.])
        fCen : float
            final cost weight for energy error
            (Default value = 0.)
        """
        self.sCu = sCu
        self.sCp = sCp
        self.sCv = sCv
        self.sCen = sCen
        self.fCp = fCp
        self.fCv = fCv
        self.fCen = fCen

    # def set_cost_parameters_(self,
    #                          pars=[0.005,  # sCu
    #                                0., 0.,        # sCp
    #                                0., 0.,        # sCv
    #                                0.,            # sCen
    #                                1000., 1000.,  # fCp
    #                                10., 10.,      # fCv
    #                                0.]):          # fCen
    #     self.sCu = [pars[0], pars[0]]
    #     self.sCp = [pars[1], pars[2]]
    #     self.sCv = [pars[3], pars[4]]
    #     self.sCen = pars[5]
    #     self.fCp = [pars[6], pars[7]]
    #     self.fCv = [pars[8], pars[9]]
    #     self.fCen = pars[10]

    def set_cost_parameters_(self,
                             pars=[0.005,  # sCu
                                   0., 0.,        # sCp
                                   0., 0.,        # sCv
                                   1000., 1000.,  # fCp
                                   10., 10.]):    # fCv
        """
        Set cost parameters used for optimization in form of a list.
        (used for parameter optimization)

        Parameters
        ----------
        pars : list
            list order=[sCu1, sCp1, sCp2, sCv1, sCv2, fCp1, fCp2, fCv1, fCv2]
            energy costs are set to 0.
            (Default value = [0.005, 0., 0., 0., 0., 1000., 1000., 10., 10.])
        """
        self.sCu = [pars[0], pars[0]]
        self.sCp = [pars[1], pars[2]]
        self.sCv = [pars[3], pars[4]]
        self.sCen = 0.0
        self.fCp = [pars[5], pars[6]]
        self.fCv = [pars[7], pars[8]]
        self.fCen = 0.0

    # def set_cost_parameters_(self,
    #                          pars=[0.005,         # sCu
    #                                0., 0.,        # sCp
    #                                0., 0.,        # sCv
    #                                0.]):          # sCen
    #     self.sCu = [pars[0], pars[0]]
    #     self.sCp = [pars[1], pars[2]]
    #     self.sCv = [pars[3], pars[4]]
    #     self.sCen = pars[5]
    #     self.fCp = [0.0, 0.0]
    #     self.fCv = [0.0, 0.0]
    #     self.fCen = 0.0

    def set_start(self, x):
        """set_start
        Set start state for the trajectory.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.start = x

    def set_goal(self, x):
        """set_goal.
        Set goal for the trajectory.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
            (Default value=[np.pi, 0., 0., 0.])
        """
        self.goal = x

    def compute_trajectory(self):
        """
        Perform the trajectory optimization calculation and return the
        trajectory.

        Returns
        -------
        numpy_array
            time points, unit=[s]
            shape=(N,)
        numpy_array
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        numpy_array
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """

        self.il = cppilqr(self.N)
        self.il.set_parameters(self.integrator_int, self.dt)
        self.il.set_start(self.start[0], self.start[1],
                          self.start[2], self.start[3])
        self.il.set_model_parameters(
            self.mass[0], self.mass[1],
            self.length[0], self.length[1],
            self.com[0], self.com[1],
            self.inertia[0], self.inertia[1],
            self.damping[0], self.damping[1],
            self.coulomb_fric[0], self.coulomb_fric[1],
            self.gravity,
            self.torque_limit[0], self.torque_limit[1])
        self.il.set_cost_parameters(self.sCu[0], self.sCu[1],
                                    self.sCp[0], self.sCp[1],
                                    self.sCv[0], self.sCv[1],
                                    self.sCen,
                                    self.fCp[0], self.fCp[1],
                                    self.fCv[0], self.fCv[1],
                                    self.fCen)
        self.il.set_goal(self.goal[0], self.goal[1],
                         self.goal[2], self.goal[3])
        # Somehow the ordering of parameter setting makes a difference
        # set_goal need to be at the end
        self.il.run_ilqr(self.max_iter,
                         self.break_cost_redu,
                         self.regu_init,
                         self.max_regu,
                         self.min_regu)

        u1_traj = self.il.get_u1_traj()
        u2_traj = self.il.get_u2_traj()
        p1_traj = self.il.get_p1_traj()
        p2_traj = self.il.get_p2_traj()
        v1_traj = self.il.get_v1_traj()
        v2_traj = self.il.get_v2_traj()

        T = np.linspace(0, self.N*self.dt, self.N)
        X = np.asarray([p1_traj, p2_traj, v1_traj, v2_traj]).T
        U = np.asarray([u1_traj, u2_traj]).T

        return T, X, U

    def save_trajectory_csv(self):
        """
        Save the trajectory to a csv file.
        The csv file will be placed in the folder where the script is executed
        and will be called "trajectory.csv".
        """
        self.il.save_trajectory_csv()
