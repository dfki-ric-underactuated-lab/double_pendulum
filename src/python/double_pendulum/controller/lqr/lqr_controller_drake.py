import os
import numpy as np
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.systems.primitives import FirstOrderTaylorApproximation
from pydrake.trajectories import PiecewisePolynomial
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.urdfs import generate_urdf


class LQRController(AbstractController):
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
        y = x.copy()
        y[0] = y[0] % (2*np.pi)
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        # set the operating point (vertical unstable equilibrium)
        self.context.get_mutable_continuous_state_vector().SetFromVector(y)
        self.xd = np.asarray(y)

    def set_cost_matrices(self, Q, R):
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
        y = x.copy()
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
