import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController


class Controller_sum_of_Gaussians_with_angles_numpy(AbstractController):
    def __init__(self, parameters, ctrl_rate, u_max=6, num_dof=2, controlled_dof=None, wait_steps=0):
        # np arrays
        self.lengthscales = np.exp(parameters['log_lengthscales'])
        self.norm_centers = parameters['centers'] / self.lengthscales
        self.linear_weight = parameters['linear_weights']
        self.u_max = u_max

        self.num_dof = num_dof
        if controlled_dof is None:
            controlled_dof = [0, 1]
        self.controlled_dof = controlled_dof

        self.ctrl_rate = ctrl_rate
        self.ctrl_cnt = 0
        self.last_control = np.zeros(self.num_dof)

        self.wait_steps = wait_steps

        super().__init__()

    def get_control_output_(self, x, t):
        meas_pos = x[:self.num_dof]
        meas_vel = x[self.num_dof:]
        if self.ctrl_cnt % self.ctrl_rate == 0 and self.ctrl_cnt >= self.wait_steps:
            state = np.zeros((self.num_dof * 3))  # velocities, cos, sin
            # print(meas_vel)
            state[:self.num_dof] = meas_vel
            state[self.num_dof:2*self.num_dof] = np.cos(meas_pos)
            state[2 * self.num_dof:] = np.sin(meas_pos)

            state = state / self.lengthscales
            dist = self.norm_centers - state
            sq_dist = np.sum(dist**2, axis=-1)
            u = np.sum(self.linear_weight * np.exp(-sq_dist))
            u = self.u_max * np.tanh((u / self.u_max))

            out_u = np.zeros(self.num_dof)
            out_u[self.controlled_dof] = u

            self.last_control = out_u

        self.ctrl_cnt += 1
        return self.last_control
