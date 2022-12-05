import os
import yaml
import numpy as np
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.systems.primitives import FirstOrderTaylorApproximation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.urdfs import generate_urdf


class LQRController(AbstractController):
    """
    LQRController.
    Controller which uses LQR to stabilize a (unstable) fixpoint.
    Uses drake LQR.

    Parameters
    ----------
    urdf_path : string or path object
        path to urdf file
    model_pars : model_parameters object
        object of the model_parameters class
    robot : string
        robot which is used, Options:
            - acrobot
            - pendubot
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    save_dir : string
        path to directory where log data can be stored
        (necessary for temporary generated urdf)
        (Default value=".")
    """
    def __init__(self,
                 urdf_path,
                 model_pars,
                 robot="acrobot",
                 torque_limit=[0.0, 1.0],
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

        self.drake_robot = MultibodyPlant(time_step=0.0)
        Parser(self.drake_robot).AddModelFromFile(self.urdf_path)
        self.drake_robot.Finalize()
        self.context = self.drake_robot.CreateDefaultContext()

        # find input and output of the drake_robot
        self.input_i = self.drake_robot.get_actuation_input_port().get_index()
        self.output_i = self.drake_robot.get_state_output_port().get_index()

        # set input of the drake_robot to zero
        self.drake_robot.get_actuation_input_port().FixValue(self.context, [0])

    def set_goal(self, x=[np.pi, 0., 0., 0.]):
        """set_goal.
        Set goal for the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
            (Default value=[np.pi, 0., 0., 0.])
        """
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        # set the operating point (vertical unstable equilibrium)
        self.context.get_mutable_continuous_state_vector().SetFromVector(y)
        self.xd = np.asarray(y)

    def set_cost_matrices(self, Q, R):
        """
        Set the Q and R matrices directly.

        Parameters
        ----------
        Q : numpy_array
            shape=(4,4)
            Q-matrix describing quadratic state cost
        R : numpy_array
            shape=(2,2)
            R-matrix describing quadratic control cost
        """
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

    def set_cost_parameters(self,
                            p1p1_cost=1.,     # 1000., 0.001
                            p2p2_cost=1.,     # 1000., 0.001
                            v1v1_cost=1.,     # 1000.
                            v2v2_cost=1.,     # 1000.
                            p1p2_cost=0.,     # -500
                            v1v2_cost=0.,     # -500
                            p1v1_cost=0.,
                            p1v2_cost=0.,
                            p2v1_cost=0.,
                            p2v2_cost=0.,
                            u1u1_cost=0.01):    # 100., 0.01
        """set_cost_parameters.
        Parameters of Q and R matrices. The parameters are

        Q = ((p1p1, p1p2, p1v1, p1v2),
             (p1p2, p2p2, p2v1, p2v2),
             (p1v1, p2v1, v1v1, v1v2),
             (p1v2, p2v2, v1v2, v2v2))
        R = ((u1u1))

        Parameters
        ----------
        p1p1_cost : float
            p1p1_cost
            (Default value=1.)
        p2p2_cost : float
            p2p2_cost
            (Default value=1.)
        v1v1_cost : float
            v1v1_cost
            (Default value=1.)
        v2v2_cost : float
            v2v2_cost
            (Default value=0.)
        p1p2_cost : float
            p1p2_cost
            (Default value=0.)
        v1v2_cost : float
            v1v2_cost
            (Default value=0.)
        p1v1_cost : float
            p1v1_cost
            (Default value=0.)
        p1v2_cost : float
            p1v2_cost
            (Default value=0.)
        p2v1_cost : float
            p2v1_cost
            (Default value=0.)
        p2v2_cost : float
            p2v2_cost
            (Default value=0.)
        u1u1_cost : float
            u1u1_cost
            (Default value=0.01)
        """
        # state cost matrix
        self.Q = np.array([[p1p1_cost, p1p2_cost, p1v1_cost, p1v2_cost],
                           [p1p2_cost, p2p2_cost, p2v1_cost, p2v2_cost],
                           [p1v1_cost, p2v1_cost, v1v1_cost, v1v2_cost],
                           [p1v2_cost, p2v2_cost, v1v2_cost, v2v2_cost]])

        # control cost matrix
        self.R = np.array([[u1u1_cost]])
        # self.R = np.array([[u2u2_cost]])

    # def set_cost_parameters_(self,
    #                          pars=[1., 1., 1., 1.,
    #                                0., 0., 0., 0., 0., 0.,
    #                                0.01, 0.01, 0.]):
    #     self.set_cost_parameters(p1p1_cost=pars[0],
    #                              p2p2_cost=pars[1],
    #                              v1v1_cost=pars[2],
    #                              v2v2_cost=pars[3],
    #                              p1v1_cost=pars[4],
    #                              p1v2_cost=pars[5],
    #                              p2v1_cost=pars[6],
    #                              p2v2_cost=pars[7],
    #                              u1u1_cost=pars[8],
    #                              u2u2_cost=pars[9],
    #                              u1u2_cost=pars[10])

    def set_cost_parameters_(self,
                             pars=[1., 1., 1., 1., 1.]):
        """
        Set the diagonal parameters of Q and R matrices with a list.

        The parameters are
        Q = ((pars[0], 0, 0, 0),
             (0, pars[1], 0, 0),
             (0, 0, pars[2], 0),
             (0, 0, 0, pars[3]))
        R = ((pars[4]))


        Parameters
        ----------
        pars : list
            shape=(5,), dtype=float
            (Default value=[1., 1., 1., 1., 1.])
        """
        self.set_cost_parameters(p1p1_cost=pars[0],
                                 p2p2_cost=pars[1],
                                 v1v1_cost=pars[2],
                                 v2v2_cost=pars[3],
                                 p1v1_cost=0.0,
                                 p1v2_cost=0.0,
                                 p2v1_cost=0.0,
                                 p2v2_cost=0.0,
                                 u1u1_cost=pars[4])

    def init_(self):
        """
        Initalize the controller.
        """
        # Linearization of the system
        self.drake_robot_lin = FirstOrderTaylorApproximation(
                self.drake_robot,
                self.context,
                input_port_index=self.input_i,
                output_port_index=self.output_i)
        self.K, self.S = LinearQuadraticRegulator(
                self.drake_robot_lin.A(),
                self.drake_robot_lin.B(),
                self.Q,
                self.R)

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
        y = x.copy().astype(float)
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi

        y -= self.xd

        tau = -self.K.dot(y)
        tau = np.asarray(tau)[0]

        u = np.zeros(2)
        u[self.active_motor] = tau

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
        return u

    def save_(self, save_dir):
        """
        Save controller parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """

        par_dict = {
                "robot" : self.robot,
                "active_motor" : self.active_motor,
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "xd1" : float(self.xd[0]),
                "xd2" : float(self.xd[1]),
                "xd3" : float(self.xd[2]),
                "xd4" : float(self.xd[3]),
        }

        with open(os.path.join(save_dir, "controller_lqr_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)

        np.savetxt(os.path.join(save_dir, "controller_lqr_Qmatrix.txt"), self.Q)
        np.savetxt(os.path.join(save_dir, "controller_lqr_Rmatrix.txt"), self.R)
        np.savetxt(os.path.join(save_dir, "controller_lqr_Kmatrix.txt"), self.K)
        np.savetxt(os.path.join(save_dir, "controller_lqr_Smatrix.txt"), self.S)
