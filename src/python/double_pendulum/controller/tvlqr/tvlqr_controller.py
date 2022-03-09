import numpy as np
import pandas as pd
from pydrake.all import (FiniteHorizonLinearQuadraticRegulatorOptions,
                         FiniteHorizonLinearQuadraticRegulator,
                         PiecewisePolynomial)
from pydrake.all import FirstOrderTaylorApproximation, LinearQuadraticRegulator
from pydrake.all import Parser, MultibodyPlant

from double_pendulum.controller.abstract_controller import AbstractController


def LQR_drake(urdf_path,
              x_star,
              Q_tilqr,
              R_tilqr):
    acrobot = MultibodyPlant(time_step=0.0)
    Parser(acrobot).AddModelFromFile(urdf_path)
    acrobot.Finalize()
    context = acrobot.CreateDefaultContext()
    # find input and output of the acrobot
    input_i = acrobot.get_actuation_input_port().get_index()
    output_i = acrobot.get_state_output_port().get_index()
    # set input of the acrobot to zero
    acrobot.get_actuation_input_port().FixValue(context, [0])
    # set the operating point (vertical unstable equilibrium)
    context.get_mutable_continuous_state_vector().SetFromVector(x_star)
    # Linearization of the system
    acrobot_lin = FirstOrderTaylorApproximation(acrobot,
                                                context,
                                                input_port_index=input_i,
                                                output_port_index=output_i)
    return LinearQuadraticRegulator(acrobot_lin.A(),
                                    acrobot_lin.B(),
                                    Q_tilqr,
                                    R_tilqr)

class TVLQRController(AbstractController):
    def __init__(self,
                 csv_path,
                 urdf_path,
                 dt=None,
                 max_t=None,
                 torque_limit=[0.0, 3.0],
                 robot="acrobot"):

        self.urdf_path = urdf_path
        #self.trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
        self.trajectory = pd.read_csv(csv_path)
        if dt is None:
            self.dt = self.trajectory["time"][1] - self.trajectory["time"][0]
        else:
            self.dt = dt
        if max_t is None:
            self.max_t = self.trajectory["time"][-1]
        else:
            self.max_t = max_t
        self.torque_limit = torque_limit
        self.robot = robot
        if self.robot == "acrobot":
            self.active_motor = 1
        else:
            self.active_motor = 0

    def set_cost_parameters(self,
                            Q=np.diag([4., 4., 0.1, 0.1]),
                            R=2*np.eye(1),
                            Qf=np.diag([11.67, 3.87, 0.1, 0.11]),
                            Rf=0.18*np.eye(1)):
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.Qf = np.asarray(Qf)
        self.Rf = np.asarray(Rf)

    def init(self):

        # trajectory loading
        self.pos1_traj = self.trajectory["shoulder_pos"]
        self.vel1_traj = self.trajectory["shoulder_vel"]
        self.tau1_traj = self.trajectory["shoulder_torque"]
        self.pos2_traj = self.trajectory["elbow_pos"]
        self.vel2_traj = self.trajectory["elbow_vel"]
        self.tau2_traj = self.trajectory["elbow_torque"]
        self.time_traj = self.trajectory["time"]

        self.time_traj = self.time_traj.values.reshape(self.time_traj.shape[0], -1)
        x0_desc = np.vstack((self.pos1_traj,
                             self.pos2_traj,
                             self.vel1_traj,
                             self.vel2_traj))

        x0 = PiecewisePolynomial.CubicShapePreserving(self.time_traj,
                                                      x0_desc,
                                                      zero_end_point_derivatives=True)
        tau_trajs = [self.tau1_traj, self.tau2_traj]
        u0_desc = tau_trajs[self.active_motor]
        u0_desc = u0_desc.values.reshape(u0_desc.shape[0], -1).T
        u0 = PiecewisePolynomial.FirstOrderHold(self.time_traj, u0_desc)

        # tvlqr construction with drake
        robot_model = MultibodyPlant(time_step=0.0)
        Parser(robot_model).AddModelFromFile(self.urdf_path)
        robot_model.Finalize()
        context = robot_model.CreateDefaultContext()
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.x0 = x0
        options.u0 = u0
        # K_tilqr, S_tilqr = LQR_drake(urdf_path=self.urdf_path,
        #                              x_star=[np.pi, 0., 0., 0.],
        #                              Q_tilqr=self.Qf,
        #                              R_tilqr=self.Rf)
        S_tilqr = np.array([[6500., 1600., 1500.,  0.],
                            [1600.,  400.,  370.,  0.],
                            [1500.,  370.,  350.,  0.],
                            [   0.,    0.,    0., 30.]])

        options.Qf = S_tilqr
        options.input_port_index = robot_model.get_actuation_input_port().get_index()
        self.tvlqr = FiniteHorizonLinearQuadraticRegulator(robot_model,
                                                           context,
                                                           t0=options.u0.start_time(),
                                                           tf=options.u0.end_time(),
                                                           Q=self.Q,
                                                           R=self.R,
                                                           options=options)

    def get_init_trajectory(self):
        T = self.time_traj
        X = np.asarray([self.pos1_traj,
                        self.pos2_traj,
                        self.vel1_traj,
                        self.vel2_traj]).T
        U = np.asarray([self.tau1_traj,
                        self.tau2_traj]).T

        return T, X, U
        

    def get_control_output(self, x, t):
        # n = int(min(t, self.max_t) / self.dt)

        error_state = np.reshape(x,(np.shape(x)[0], 1)) - self.tvlqr.x0.value(t)

        tau = (self.tvlqr.u0.value(t) -
               self.tvlqr.K.value(t).dot(error_state) -
               self.tvlqr.k0.value(t))[0][0]

        u = np.zeros(2)
        u[self.active_motor] = tau

        u[0] = np.clip(u[0], -self.torque_limit[0], self.torque_limit[0])
        u[1] = np.clip(u[1], -self.torque_limit[1], self.torque_limit[1])
        return u
