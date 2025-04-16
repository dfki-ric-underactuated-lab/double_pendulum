import os
import time
import numpy as np
import torch
import math
from double_pendulum.controller.abstract_controller import AbstractController
from .utils.models import load_model
from .utils.normalization import LimitsNormalizer
import double_pendulum.controller.prx.prx_utils as prx 

MODEL_DIR_PATH = "./models"
MODEL_NAME = "25_03_14-21_02_37_H_PADF_HIST_PADF_LD0.99_transformer_large"

class AcrobotFlowMatchingController(AbstractController):

    def __init__(self, model_path, device="cpu", horizon_length=None):
        super().__init__()
        self.model, self.model_args = load_model(model_path, device)
        self.normalizer = LimitsNormalizer(params=self.model_args.normalization_params)
        self.history_length = self.model_args.history_length
        self.horizon_length = horizon_length if horizon_length is not None else self.model_args.horizon_length
        self.stride = self.model_args.history_stride
        self.action_index = self.model_args.action_indices[0]

        self.history_buffer = torch.zeros((self.history_length, 5))
        
        self.stride_counter = 0
        self.prev_t = 0
        self.horizon_left = 0
        self.torque_limit = 6
        
        self.u = None
        self.predicted_horizon = None

        self.is_transformer = "Transformer" in self.model_args.model
        self.use_fm = True
        # LQR
        self.K = np.matrix([[-255.751, -107.574, -54.1521, -24.8681]]);
        self.goal = np.matrix([math.pi,0.0, 0.0, 0.0]).reshape((4,1))
        self.zero = np.matrix([0.0, 0.0, 0.0, 0.0]).reshape((4,1))
        self.lqr_time=0
        self.prev_t = 0.0
        self.device = device



    def update_history_buffer(self, x):
        # Shift history buffer up, removing first entry
        self.history_buffer = self.history_buffer.roll(-1, dims=0)
        # Update last entry with new state
        self.history_buffer[-1, :4] = torch.tensor(x, dtype=torch.float32, device=self.device)
        if self.u is not None:
            self.history_buffer[-1, 4] = torch.tensor(self.u, dtype=torch.float32, device=self.device)

    def get_conditions(self):
        if self.is_transformer:
            cond = {}
            for i in range(self.history_length):
                cond[i] = self.history_buffer[i]
        else:
            cond = {}
            for i in range(self.history_length):
                cond[i] = self.history_buffer[i].unsqueeze(0)

        return cond

    def compute_new_control(self, x):
        cond = self.get_conditions()
        
        sample = self.model.conditional_sample(cond)

        self.predicted_horizon = sample.horizon

    def update_control(self, x):
        self.update_history_buffer(x)
    
        if self.horizon_left == 0:
            self.compute_new_control(x)
            self.horizon_left = self.horizon_length

        u = self.predicted_horizon[self.horizon_length - self.horizon_left, self.action_index]
        self.u = np.clip(u.item(), -self.torque_limit, self.torque_limit)

        self.horizon_left -= 1


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
        # dt = t - self.prev_t;
        # goal_err = prx.compute_state_diff(x.reshape((4,1)), self.goal).reshape((4,1));
        # if dt > 0.01:
        #     self.use_fm = True
        # else:
        #     th_err = np.linalg.norm(goal_err[0:2])
        #     if self.lqr_time > 1.0 and th_err > 0.2:
        #         self.use_fm = True
        # if self.use_fm and math.fabs(goal_err[0]) < 0.5 and goal_err[2] + goal_err[3] < 5:
        #     self.lqr_time = 0
        #     self.use_fm = False
        # if math.fabs(goal_err[2]) > 25 or  math.fabs(goal_err[3]) > 25:
        #     self.use_fm = True

        # if self.use_fm:
        #     if self.stride_counter % self.stride == 0:
        #         self.update_control(x)
        # else: # LQR
        #     self.u = prx.compute_control_from_lqr(x, self.K, self.goal);
        #     self.lqr_time += dt

        if self.stride_counter % self.stride == 0:
            self.update_control(x)
        u = [0.0, self.u]

        self.stride_counter += 1
        return u