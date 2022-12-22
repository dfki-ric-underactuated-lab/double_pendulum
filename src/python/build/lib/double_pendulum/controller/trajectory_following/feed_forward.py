import os
import yaml
import numpy as np

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.utils.pcw_polynomial import FitPiecewisePolynomial, InterpolateVector


class FeedForwardController(AbstractController):
    """FeedforwardController
    Controller which executes a feedforward torque trajectory

    Parameters
    ----------
    T : array_like
        shape=(N,)
        time points of reference trajectory, unit=[s]
        (Default value=None)
    U : array_like
        shape=(N, 2)
        reference trajectory actuations/motor torques
        order=[u1, u2],
        units=[Nm]
        (Default value=None)
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[1.0, 1.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    num_break : int
        number of break points used for interpolation
        (Default value = 40)
    """
    def __init__(self,
                 T,
                 U,
                 torque_limit=[1.0, 1.0],
                 num_break=40):

        super().__init__()

        self.T = T
        self.U = U
        self.torque_limit = torque_limit
        self.num_break = num_break

        self.U_interp = InterpolateVector(
                T=self.T,
                X=self.U,
                num_break=num_break,
                poly_degree=3)

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
        tt = min(t, self.T[-1])

        uu = self.U_interp.get_value(tt)
        u1 = np.clip(uu[0], -self.torque_limit[0], self.torque_limit[0])
        u2 = np.clip(uu[1], -self.torque_limit[1], self.torque_limit[1])

        u = np.asarray([u1, u2])
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
                "torque_limit1" : self.torque_limit[0],
                "torque_limit2" : self.torque_limit[1],
                "num_break" : self.num_break,
        }

        with open(os.path.join(save_dir, "controller_feedforward_parameters.yml"), 'w') as f:
            yaml.dump(par_dict, f)
