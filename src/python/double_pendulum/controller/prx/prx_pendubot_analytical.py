import os
import yaml
import numpy as np
import math
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.gravity_compensation.gravity_compensation_controller import GravityCompensationController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.prx.prx_utils import lqr_traj_follower
import double_pendulum.controller.prx.prx_utils as prx 
# from double_pendulum.controller.prx_utils import AbstractController

class PrxPendubotAnalyticalController(AbstractController):
    """
    LQRController.
    Controller which uses LQR to stabilize a (unstable) fixpoint.

    Parameters
    ----------
    """
 
    def __init__(self, filename):
        super().__init__()
        self.ilqr = prx.lqr_traj_follower(filename)
        # self.traj_lqr_from_file(filename)
        self.grav_con = GravityCompensationController(
                mass=[0.5234602302310271, 0.6255677234174437],
                length=[0.2, 0.3],
                com=[ 0.2, 0.25569305436052964],
                damping=[0.0, 0.0],
                coulomb_fric=[0.0, 0.0],
                gravity=9.81,
                inertia=[0.031887199591513114, 0.05086984812807257],
                torque_limit=[6.0, 0.0])
        self.pid_con = PointPIDController(
                torque_limit=[6.0, 0.0],
                dt=0.002)


    def init_(self):
        """
        Initalize the controller.
        """
        self.K = np.matrix([[-57.2535, -54.4997, -13.3048, -9.66839]]);
        # self.K0 = np.matrix([[-44.0505, -15.4619, -3.48382, 9.42782]]);
        # self.K0 = np.matrix([[-46.8337, -9.29643, -2.17849, 10.0649]]);
        self.K0 = np.matrix([[-0.760964, -0.765396, 0.340315, -0.17986]]);
        self.goal = np.matrix([math.pi,0.0, 0.0, 0.0]).reshape((4,1))
        self.zero = np.matrix([0,0,0,0]).reshape((4,1))
        # self.zero = np.matrix([0,0,0,0]).reshape((4,1))
        self.u = 0.0;
        self.torque_limit = 6
        self.idx = 0
        self.prev_t = 0.0
        self.grav_con.init()
        self.use_gc = False
        self.pid_con.set_parameters(10.0, 0.1, 0.1)
        self.pid_con.set_goal(self.zero)
        self.pid_con.init()
        self.lqr_time=0
        self.zero_done = False
        self.side = 1
    
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

        dt = t - self.prev_t;

        goal_err = prx.compute_state_diff(x.reshape((4,1)), self.goal).reshape((4,1));
        th_err = np.linalg.norm(goal_err[0:2])
        if dt > 0.01:
            self.use_gc = True
            # min_idx, err = self.ilqr.find_closest_idx(x)
            # print("DISTURBANCE")
        else:
            if self.lqr_time > 1.0 and th_err > 0.2:
                self.use_gc = True

        if self.use_gc and math.fabs(goal_err[0]) < 0.5 and goal_err[2] + goal_err[3] < 5:
            # print("Back to LQR", goal_err, math.fabs(goal_err[0]), math.fabs(goal_err[1]), math.fabs(goal_err[2]), math.fabs(goal_err[3]))
            self.lqr_time = 0
            self.use_gc = False

        if math.fabs(goal_err[2]) > 25 or  math.fabs(goal_err[3]) > 25:
            self.use_gc = True


        if self.use_gc:
            self.lqr_time = 0

            # min_idx, err = self.ilqr.find_closest_idx(x)
            # min_idx, err = self.ilqr.find_closest_idx(x)
            xerr = prx.compute_state_diff(x.reshape((4,1)), self.zero).reshape((4,1));
            # err = np.linalg.norm(xerr)
            err = xerr.T @ np.diag([1,1,0.1,0.1]) @ xerr

            # self.compute_control_from_traj(x)
            # print(t, x, xerr.flatten(), err)
            # if err < .10:
            # if math.fabs(xerr[0]) < 0.2 and math.fabs(xerr[1]) < 0.2 and math.fabs(xerr[2]) < 0.5 and math.fabs(xerr[3]) < 0.5:
            min_idx, err = self.ilqr.find_closest_idx(x)
            min_idx_m, err_m = self.ilqr.find_closest_idx(-x)
                # print("Minus", t, xerr, err, err_m)
            # if err < .80:
            # threshold=1.5
            # threshold=0.8
            threshold=0.7
            # threshold=0.6
            if err < threshold or err_m < threshold:
                # print(t, xerr, err, err_m)
                if err_m < threshold:
                    self.side = -1
                    self.ilqr.idx = min_idx_m
                    # print("Back to Traj Min", t, x, err_m, min_idx_m)
                else: 
                    self.side = 1
                    self.ilqr.idx = min_idx
                    # print("Back to Traj Max", t, x, err,min_idx)

                # min_idx, err = self.ilqr.find_closest_idx(x)
                # print("changing to traj", t, err, min_idx)
                # self.idx = 0
                self.use_gc = False
                # self.zero_done = True
                # self.ilqr.idx = 0
                self.u = self.side * self.ilqr.compute_control_from_traj(self.side * x)
                self.ilqr.idx += 1

            # if self.zero_done:                
            #     k = np.array([[4660.25, 6076.72, 509.651, 1083.64]]);
            #     lg = np.array([0.0,3.1415926536,0,0]);
            #     self.u = prx.compute_control_from_lqr(x, k, lg);

            else:
                # print(x, t)
                # grav_u = self.grav_con.get_control_output(x, t)
                # pid_u = self.pid_con.get_control_output(x, t)
                # u = grav_u + pid_u
                # u = grav_u
                # self.u[1] = u[1]
                self.u = prx.compute_control_from_lqr(x, self.K0, self.zero);
                # print("LQR to zero", t, x, self.u)
        else:
            # if self.idx < len(self.gains):
            if self.ilqr.valid():
                # print("ilqr")
                # min_idx, _ = self.ilqr.find_closest_idx(x)
                # self.ilqr.idx = min_idx

                self.u = self.side * self.ilqr.compute_control_from_traj(self.side * x)
                self.ilqr.idx += 1
                # print("iLQR", t, x)
                # self.compute_control_from_traj(x)
                # self.idx += 1
            else:
                # print("LQR")
                self.u = prx.compute_control_from_lqr(x, self.K, self.goal);
                self.lqr_time += dt
                # print("LQR to Goal", t, x)

        self.prev_t = t
        self.u = np.clip(self.u, -self.torque_limit, self.torque_limit)

        return [self.u, 0.0]

