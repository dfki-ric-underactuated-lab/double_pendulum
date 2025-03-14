import os
import yaml
import numpy as np
import math

def compute_angle_diff(diff):
    return np.arctan2(np.sin(diff), np.cos(diff))

def compute_state_diff(x, goal):
    xp = x - goal
    xp[0:2] = compute_angle_diff(xp[0:2])
    return xp

def compute_state_diff_2(x, goal):
    xp = x - goal
    xp[:, 0:2] = compute_angle_diff(xp[:, 0:2])
    return xp

def compute_control_from_lqr(x, K, goal):
    xp = compute_state_diff(x.reshape((4,1)), goal)
    torque = np.matmul(-K, xp)
    return torque[0,0]

class lqr_traj_follower():
    def __init__(self, filename):
        self.traj_lqr_from_file(filename)
        self.idx = 0

    def traj_lqr_from_file(self, filename):
        file = open(filename, 'r')
        self.states = []
        self.controls = []
        self.gains = []
        for line in file:
            arr = line.split();
            # print(arr)
            if len(arr) > 5:
                self.states.append(np.asarray(arr[0:4], dtype=np.float64).reshape((4,1)))
                self.controls.append(np.asarray(arr[4], dtype=np.float64))
                self.gains.append(np.asarray(arr[5:], dtype=np.float64))
        self.states_np = np.array(self.states).reshape((-1,4));
        # print(self.states_np.shape)
        # print(self.states_np)
    
    def valid(self):
        return self.idx < len(self.gains)

    def compute_control_from_traj(self, x):
        xp = compute_state_diff(x.reshape((4,1)), self.states[self.idx])
        u = self.controls[self.idx];
        du = -self.gains[self.idx] @ xp
        # print(xp, xp.shape, du, du.shape)
        return u + du[0]
        # print(u, du, self.u)

    def find_closest_idx(self, x):
        xp = compute_state_diff_2(x.reshape((1,4)), self.states_np)
            
        xp = np.linalg.norm(xp, axis=1)
        # print(xp)
        min_idx = np.argmin(xp);
        return min_idx, xp[min_idx]
