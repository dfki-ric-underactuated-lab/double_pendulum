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


class Controller_muli_out_sum_of_Gaussians_with_angles_numpy(AbstractController):
    def __init__(self, parameters, ctrl_rate, u_max=None, num_dof=2, controlled_dof=None, wait_steps=0):
        # np arrays
        if u_max is None:
            u_max = [5.]
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
        self.parameters = parameters
        self.wait_steps = wait_steps

        super().__init__()

    def get_control_output_(self, x, t=None):
        meas_pos = x[:self.num_dof]
        meas_vel = x[self.num_dof:]

        if self.ctrl_cnt % self.ctrl_rate == 0 and self.ctrl_cnt >= self.wait_steps:
            x = np.zeros((self.num_dof * 3))  # velocities, cos, sin
            # print(meas_vel)
            x[:self.num_dof] = meas_vel
            x[self.num_dof:2 * self.num_dof] = np.cos(meas_pos)
            x[2 * self.num_dof:] = np.sin(meas_pos)

            x = x / self.lengthscales
            dist = self.norm_centers - x
            sq_dist = np.sum(dist ** 2, axis=-1)

            # print(self.u_max)

            # u = np.zeros(len(self.u_max))
            u = np.sum(self.linear_weight * np.exp(-sq_dist), axis=1)

            for i in range(len(self.u_max)):
                # u[i] = np.sum(self.linear_weight[i, :] * np.exp(-sq_dist))
                u[i] = self.u_max[i] * np.tanh((u[i] / self.u_max[i]))

            # print(u)

            out_u = np.zeros(len(self.controlled_dof))
            out_u[self.controlled_dof] = u

            self.last_control = out_u

        self.ctrl_cnt += 1
        return self.last_control

    def get_np_policy(self, ret_dict=False):
        if ret_dict:
            return self, self.parameters
        return self


class Controller_multi_policy_sum_of_gaussians_with_angles_numpy(AbstractController):
    def __init__(self, parameters_list, ctrl_rate, u_max=[5.], num_dof=2, controlled_dof=None, active_pos_list=None,
                 active_vel_list=None, wait_steps=0):
        # list of np arrays
        self.lengthscales = []
        self.norm_centers = []
        self.linear_weight = []
        self.parameters_list = parameters_list

        for parameters in self.parameters_list:
            self.lengthscales.append(np.exp(parameters['log_lengthscales']))
            self.norm_centers.append(parameters['centers'] / np.exp(parameters['log_lengthscales']))
            self.linear_weight.append(parameters['linear_weights'])

        # list
        self.u_max = u_max

        self.num_dof = num_dof
        if controlled_dof is None:
            controlled_dof = [0, 1]
        self.controlled_dof = controlled_dof

        self.ctrl_rate = ctrl_rate
        self.ctrl_cnt = 0
        self.parameters = parameters

        if active_pos_list is None:
            active_pos_list = [list(range(num_dof))] * num_dof
        self.active_pos_list = active_pos_list

        if active_vel_list is None:
            active_vel_list = [list(range(num_dof, 2*num_dof))] * num_dof
        self.active_vel_list = active_vel_list

        self.last_control = np.zeros(self.num_dof)

        self.wait_steps = wait_steps

        super().__init__()

    def get_control_output_(self, x, t=None):

        if self.ctrl_cnt % self.ctrl_rate == 0 and self.ctrl_cnt >= self.wait_steps:
            # x = np.zeros((self.num_dof * 3))  # velocities, cos, sin
            # print(meas_vel)
            # x[:self.num_dof] = meas_vel
            # x[self.num_dof:2 * self.num_dof] = np.cos(meas_pos)
            # x[2 * self.num_dof:] = np.sin(meas_pos)

            out_u = np.zeros(self.num_dof)  # as many outputs as policies

            for i in range(self.num_dof):
                meas_pos = x[self.active_pos_list[i]]
                meas_vel = x[self.active_vel_list[i]]
                x_ = np.concatenate([meas_vel, np.cos(meas_pos), np.sin(meas_pos)])
                # print(i, x_)

                x_ = x_ / self.lengthscales[i]
                dist_ = self.norm_centers[i] - x_
                dist_ = dist_.reshape((-1, self.lengthscales[i].size))

                # print(i, dist_)

                sq_dist = np.sum(dist_ ** 2, axis=-1)

                # print(i, sq_dist)

                u = np.sum(self.linear_weight[i] * np.exp(-sq_dist))
                # print(i, u)
                out_u[i] = self.u_max[i] * np.tanh((u / self.u_max[i]))

            # print(out_u)

            self.last_control = out_u[self.controlled_dof]

        self.ctrl_cnt += 1
        return self.last_control


    def get_np_policy(self, ret_dict=False):
        if ret_dict:
            return self, self.parameters_list
        return self
