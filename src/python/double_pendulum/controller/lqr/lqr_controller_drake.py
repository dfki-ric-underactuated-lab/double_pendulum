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
