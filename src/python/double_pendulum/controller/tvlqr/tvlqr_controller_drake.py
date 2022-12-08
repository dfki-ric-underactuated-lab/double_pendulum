import os
import yaml
import numpy as np
from pydrake.systems.controllers import (FiniteHorizonLinearQuadraticRegulatorOptions,
                                         FiniteHorizonLinearQuadraticRegulator)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.csv_trajectory import load_trajectory, save_trajectory
from double_pendulum.utils.urdfs import generate_urdf

#from pydrake.systems.primitives import FirstOrderTaylorApproximation
#from pydrake... import LinearQuadraticRegulator
# def LQR_drake(urdf_path,
#               x_star,
#               Q_tilqr,
#               R_tilqr):
#     acrobot = MultibodyPlant(time_step=0.0)
#     Parser(acrobot).AddModelFromFile(urdf_path)
#     acrobot.Finalize()
#     context = acrobot.CreateDefaultContext()
#     # find input and output of the acrobot
#     input_i = acrobot.get_actuation_input_port().get_index()
#     output_i = acrobot.get_state_output_port().get_index()
#     # set input of the acrobot to zero
#     acrobot.get_actuation_input_port().FixValue(context, [0])
#     # set the operating point (vertical unstable equilibrium)
#     context.get_mutable_continuous_state_vector().SetFromVector(x_star)
#     # Linearization of the system
#     acrobot_lin = FirstOrderTaylorApproximation(acrobot,
#                                                 context,
#                                                 input_port_index=input_i,
#                                                 output_port_index=output_i)
#     return LinearQuadraticRegulator(acrobot_lin.A(),
#                                     acrobot_lin.B(),
#                                     Q_tilqr,
#                                     R_tilqr)
# K_tilqr, S_tilqr = LQR_drake(urdf_path=self.urdf_path,
#                              x_star=[np.pi, 0., 0., 0.],
#                              Q_tilqr=self.Qf,
#                              R_tilqr=self.Rf)


class TVLQRController(AbstractController):
    """TVLQRController
    Controller to stabilize a trajectory with TVLQR
    Uses the TVLQR controller from drake.

    Parameters
    ----------
    csv_path : string or path object
        path to csv file where the trajectory is stored.
        csv file should use standarf formatting used in this repo.
        If T, X, or U are provided they are preferred.
        (Default value="")
    urdf_path : string or path object
        path to urdf file
    model_pars : model_parameters object
        object of the model_parameters class
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 3.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    robot : string
        robot which is used, Options:
            - "acrobot"
            - "pendubot"
    save_dir : string
        path to directory where log data can be stored
        (necessary for temporary generated urdf)
        (Default value=".")
    """
    def __init__(self,
                 csv_path,
                 urdf_path,
                 model_pars,
                 torque_limit=[0.0, 3.0],
                 robot="acrobot",
                 save_dir="."):

        super().__init__()

        self.urdf_path = os.path.join(save_dir, robot + ".urdf")
        generate_urdf(urdf_path, self.urdf_path, model_pars=model_pars)

        self.torque_limit = torque_limit
        self.robot = robot
        if self.robot == "acrobot":
            self.active_motor = 1
        else:
            self.active_motor = 0

        T, X, U = load_trajectory(csv_path=csv_path,
                                  with_tau=True)

        self.time_traj = T
        self.pos1_traj = X.T[0]
        self.pos2_traj = X.T[1]
        self.vel1_traj = X.T[2]
        self.vel2_traj = X.T[3]
        self.tau1_traj = U.T[0]
        self.tau2_traj = U.T[1]

        self.max_t = self.time_traj[-1]

        # create drake robot model
        self.robot_model = MultibodyPlant(time_step=0.0)
        Parser(self.robot_model).AddModelFromFile(self.urdf_path)
        self.robot_model.Finalize()
        self.context = self.robot_model.CreateDefaultContext()

        # set default parameters
        self.set_cost_parameters()

    def set_cost_parameters(self,
                            Q=np.diag([4., 4., 0.1, 0.1]),
                            R=2*np.eye(1),
                            Qf=np.zeros((4, 4))):
        """set_cost_parameters
        Set the cost matrices Q, R and Qf.
        (Qf for the final stabilization)

        Parameters
        ----------
        Q : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
            (Default value=np.diag([4., 4., 0.1, 0.1]))
        R : numpy_array
            shape=(2,2)
            R-matrix describing quadratic control cost
            (Default value=2*np.eye(1))
        Qf : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
            for the final point stabilization
            (Default value=np.zeros((4, 4)))
        """

        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.Qf = np.asarray(Qf)

    def init_(self):
        """
        Initalize the controller.
        """
        # self.time_traj = self.time_traj.reshape(self.time_traj.shape[0], -1)
        x0_desc = np.vstack((self.pos1_traj,
                             self.pos2_traj,
                             self.vel1_traj,
                             self.vel2_traj))
        x0 = PiecewisePolynomial.CubicShapePreserving(self.time_traj,
                                                      x0_desc,
                                                      zero_end_point_derivatives=True)

        tau_trajs = [self.tau1_traj, self.tau2_traj]
        u0_desc = tau_trajs[self.active_motor]
        u0_desc = u0_desc.reshape(u0_desc.shape[0], -1).T
        u0 = PiecewisePolynomial.FirstOrderHold(self.time_traj[:np.shape(u0_desc)[-1]], u0_desc)

        # tvlqr construction with drake
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.x0 = x0
        options.u0 = u0
        options.Qf = self.Qf
        options.input_port_index = self.robot_model.get_actuation_input_port().get_index()
        self.tvlqr = FiniteHorizonLinearQuadraticRegulator(self.robot_model,
                                                           self.context,
                                                           t0=options.u0.start_time(),
                                                           tf=options.u0.end_time(),
                                                           Q=self.Q,
                                                           R=self.R,
                                                           options=options)

    def get_init_trajectory(self):
        """
        Get the initial (reference) trajectory used by the controller.

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
        T = self.time_traj
        X = np.asarray([self.pos1_traj,
                        self.pos2_traj,
                        self.vel1_traj,
                        self.vel2_traj]).T
        U = np.asarray([self.tau1_traj,
                        self.tau2_traj]).T

        return T, X, U

    def get_control_output_(self, x, t):
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
        # ti = float(np.min(t, self.max_t))

        error_state = np.reshape(x, (np.shape(x)[0], 1)) - self.tvlqr.x0.value(t)

        # tau = (self.tvlqr.u0.value(t) -
        #        self.tvlqr.K.value(t).dot(error_state) -
        #        self.tvlqr.k0.value(t))[0][0]
        tau = (self.tvlqr.u0.value(t) -
               self.tvlqr.K.value(t).dot(error_state))[0][0]

        # print(self.tvlqr.u0.value(t), self.tvlqr.K.value(t).dot(error_state), self.tvlqr.k0.value(t))
        # print(self.tvlqr.K.value(t))

        u = np.zeros(2)
        u[self.active_motor] = tau

        #print(t, self.tvlqr.x0.value(t).T, self.tvlqr.u0.value(t), self.tvlqr.K.value(t), u[1])
        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
        return u

    def save_(self, save_dir):
        """
        Save the energy trajectory to file.

        Parameters
        ----------
        path : string or path object
            directory where the parameters will be saved
        """

        par_dict = {
                "robot" : self.robot,
                "active_motor" : self.active_motor,
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "max_t" : self.max_t,
        }

        with open(os.path.join(save_dir, "controller_tvlqr_drake_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)

        np.savetxt(os.path.join(save_dir, "controller_tvlqr_drake_Qmatrix.txt"), self.Q)
        np.savetxt(os.path.join(save_dir, "controller_tvlqr_drake_Rmatrix.txt"), self.R)
        np.savetxt(os.path.join(save_dir, "controller_tvlqr_drake_Qfmatrix.txt"), self.Qf)

        T, X, U = self.get_init_trajectory()
        save_trajectory(os.path.join(save_dir, "initial_tvlqr_drake_traj.csv"), T, X, U)

        if not os.path.exists(os.path.join(save_dir, self.robot+".urdf")):
            os.system(f"cp {self.urdf_path} " + os.path.join(save_dir, self.robot+".urdf"))
