# here, the abstract filter class is defined, to which all filter
# classes have to adhere
import os
import yaml
from abc import ABC, abstractmethod
import numpy as np


class AbstractFilter(ABC):
    """
    Abstract filter class. All filter should inherit from
    this abstract class.
    """

    def __init__(self):
        self.x_hist = []
        self.u_hist = [[0.0, 0.0]]
        self.x_filt_hist = []
        pass

    @abstractmethod
    def get_filtered_state_(self, x, u, t=None):
        """
        Filter the state x and return filtered value.
        This function has to be overwritten by actual filter.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        u : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        t : double
            time,
            unit=s

        Returns
        -------
        numpy_array
            shape=(4,), dtype=float,
            filters state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        return x

    def get_filtered_state(self, x, u, t=None):
        """
        The method which is called in the Simulator and real experiment loop
        to get the control signal from the controller.

        Parameters
        ----------
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        u : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        t : double
            time,
            unit=s

        Returns
        -------
        numpy_array
            shape=(4,), dtype=float,
            filters state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """

        self.x_hist.append(x)
        self.u_hist.append(u)

        y = self.get_filtered_state_(x, u, t)

        self.x_filt_hist.append(y)
        return y

    def init_(self):
        """
        Initialize the filter. Optional.
        Can be overwritten by actual filter.
        Initialize function which will always be called before using the
        filter.
        """
        pass

    def init(self):
        """
        Initialize function which will always be called before using the
        filter.
        In addition to the filter specific init_ method this method
        initialized the filter and internal logs.
        """
        self.x_hist = []
        self.u_hist = [[0.0, 0.0]]
        self.xfilt_hist = []

        self.init_()

    def save(self, save_dir):
        """
        Save filter parameters

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        self.save_(save_dir)

    def save_(self, save_dir):
        """
        Save filter parameters. Optional
        Can be overwritten by actual filter.

        Parameters
        ----------
        save_dir : string or path object
            directory where the parameters will be saved
        """
        pass
