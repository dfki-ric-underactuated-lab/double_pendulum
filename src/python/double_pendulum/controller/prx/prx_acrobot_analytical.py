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

class PrxAcrobotAnalyticalController(AbstractController):
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
                torque_limit=[0.0, 6.0])
        self.pid_con = PointPIDController(
                torque_limit=[0.0, 6.0],
                dt=0.002)

 
    def init_(self):
        """
        Initalize the controller.
        """
        # self.K = np.matrix([[-255.751, -107.574, -54.1521, -24.8681]]);
        self.K = np.matrix([[-193.454, -79.8894, -40.9552, -18.5224]])
        # self.K0 = np.matrix([[-44.0505, -15.4619, -3.48382, 9.42782]]);
        # self.K0 = np.matrix([[-46.8337, -9.29643, -2.17849, 10.0649]]);
        self.K0 = np.matrix([[-0.254049, -0.103628, 0.0291697,  0.126939]]);
        self.goal = np.matrix([math.pi,0.0, 0.0, 0.0]).reshape((4,1))
        self.zero = np.matrix([0.0, 0.0, 0.0, 0.0]).reshape((4,1))
        self.u = 0.0;
        self.torque_limit = 6
        self.idx = 0
        self.prev_t = 0.0
        self.grav_con.init()
        self.pid_con.init()
        self.use_gc = False
        self.pid_con.set_goal(self.zero)
        self.pid_con.set_parameters(1.0, 0.01, 0.1)
        self.lqr_time=0
        self.side = 1
        self.ilqr.idx = 0 
        
    # def compute_angle_diff(self, diff):
    #     return np.arctan2(np.sin(diff), np.cos(diff))


    # def compute_state_diff(self, x, goal):
    #     xp = x - goal

    #     # print(xp)
    #     xp[0:2] = self.compute_angle_diff(xp[0:2])
    #     # print(xp)
    #     # xp[1] = self.compute_angle_diff(xp[])
    #     return xp

    # def compute_state_diff_2(self, x, goal):
    #     xp = x - goal

    #     # print(xp)
    #     xp[:, 0:2] = self.compute_angle_diff(xp[:, 0:2])
    #     # print(xp)
    #     # xp[1] = self.compute_angle_diff(xp[])
    #     return xp

    # def compute_control_from_lqr(self, x, K, goal):
    #     xp = self.compute_state_diff(x.reshape((4,1)), goal)
    #     torque = np.matmul(-K, xp)
    #     self.u[1] = torque[0,0]

    # def compute_control_from_traj(self, x):

    #     xp = self.compute_state_diff(x.reshape((4,1)), self.states[self.idx])
    #     u = self.controls[self.idx];
    #     du = -self.gains[self.idx] @ xp
    #     # print(xp, xp.shape, du, du.shape)
    #     self.u[1] = u + du[0]
    #     # print(u, du, self.u)

    # def find_closest_idx(self, x):
    #     xp = compute_state_diff_2(x.reshape((1,4)), self.states_np)
            
    #     xp = np.linalg.norm(xp, axis=1)
    #     # print(xp)
    #     min_idx = np.argmin(xp);
    #     return min_idx, xp[min_idx]

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
        else:
            if self.lqr_time > 1.0 and th_err > 0.2:
                self.use_gc = True
        if self.use_gc and math.fabs(goal_err[0]) < 0.5 and goal_err[2] + goal_err[3] < 5:
            self.lqr_time = 0
            self.use_gc = False
        if math.fabs(goal_err[2]) > 25 or  math.fabs(goal_err[3]) > 25:
            self.use_gc = True
              
        # goal_err = prx.compute_state_diff(x.reshape((4,1)), self.goal);
        # goal_err = np.linalg.norm(goal_err, axis=1)
        # if self.lqr_time > 2.0 and goal_err[0] > 0.1:
        #     self.use_gc = True
        # # if self.ilqr.valid() and:

        if self.use_gc:
            self.lqr_time = 0
            # min_idx, err = self.ilqr.find_closest_idx(x)
            # err = prx.compute_state_diff(x.reshape((4,1)), self.zero);
            # err = xerr.T @ np.diag([1,1,0.1,0.1]) @ xerr
            # print(err)
            # err = math.sqrt(err[0]*err[0] + err[1]*err[1])
            
            min_idx, err = self.ilqr.find_closest_idx(x)
            min_idx_m, err_m = self.ilqr.find_closest_idx(-x)

            threshold=2.0
            # if err < threshold:
            if err < threshold or err_m < threshold:
                # print("Back to Traj Min", t, x, err_m, min_idx_m)
                if err_m < threshold:
                    self.side = -1
                    self.ilqr.idx = min_idx_m
                else: 
                    self.side = 1
                    self.ilqr.idx = min_idx
                self.use_gc = False
                self.u = self.side * self.ilqr.compute_control_from_traj(self.side * x)
                self.ilqr.idx += 1
                # self.get_control_output_(x, self.prev_t)

            else:
                self.u = prx.compute_control_from_lqr(x, self.K0, self.zero);
        else:
            # if self.idx < len(self.gains):
            if self.ilqr.valid():
                # print("ilqr")
                self.u = self.ilqr.compute_control_from_traj(x)
                self.ilqr.idx += 1
                # self.compute_control_from_traj(x)
                # self.idx += 1
            else:
                self.u = prx.compute_control_from_lqr(x, self.K, self.goal);
                self.lqr_time += dt

        self.prev_t = t
        self.u = np.clip(self.u, -self.torque_limit, self.torque_limit)
        # self.u[1] = np.clip(self.u[1], -self.torque_limit, self.torque_limit)
        return [0.0, self.u]
        # return self.u

