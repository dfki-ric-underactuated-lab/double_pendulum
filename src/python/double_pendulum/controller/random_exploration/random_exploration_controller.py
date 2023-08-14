import numpy as np
from double_pendulum.controller.abstract_controller import AbstractController

import matplotlib.pyplot as plt
from scipy.signal import butter
import scipy.signal as signal

class Controller_Random_exploration(AbstractController):
    def __init__(self, ctrl_rate, filt_freq, seed, type_random='WGN', random_par=None, expl_time=10., system_freq=500,
                 u_max=6, num_dof=2, controlled_dof=None, plot_profile=True, wait_steps=0):
        """
            type_random:
                -WGN: white Gaussian noise
                -SUM_SIN: sum of sinusoids
        """
        self.u_max = u_max
        self.num_dof = num_dof
        if controlled_dof is None:
            controlled_dof = list(range(self.num_dof))
        self.controlled_dof = controlled_dof

        self.ctrl_cnt = 0
        self.last_control = np.zeros(self.num_dof)

        self.ctrl_rate = ctrl_rate
        self.filt_freq = filt_freq

        self.seed = seed
        np.random.seed(self.seed)

        self.type_random = type_random
        self.random_par = random_par

        if self.type_random == 'WGN':
            if self.random_par is None:
                self.random_par = {'std': self.u_max / 3, 'butter_order': 2}
        elif self.type_random == 'SUM_SIN':
            if random_par is None:
                self.random_par = {'sin_freq': 10, 'num_sin': 5, 'butter_order': 2}
        else:
            raise NotImplementedError('Random type {} not implemented'.format(self.type_random))

        self.expl_time = expl_time
        self.system_freq = system_freq

        self.init_profile(plot_profile=plot_profile)

        self.wait_steps = wait_steps

        super().__init__()

    def init_profile(self, plot_profile=True):
        self.u_profile = None
        n_samples = int(self.expl_time * self.system_freq)
        self.u_profile = np.zeros((n_samples, self.num_dof))
        resamp = np.zeros((int(n_samples/self.ctrl_rate), self.num_dof))
        b, a = butter(self.random_par['butter_order'], self.filt_freq, analog=False, fs=self.system_freq)
        sys_time = np.arange(0, self.expl_time, 1 / self.system_freq)

        if self.type_random == 'WGN':
            std = self.random_par['std']
            print(std)
            for k in self.controlled_dof:
                self.u_profile[:, k] = std * np.random.normal(np.zeros((n_samples, )),
                                                              np.ones((n_samples, )))
        elif self.type_random == 'SUM_SIN':
            num_sin = self.random_par['num_sin']
            sin_freq = self.random_par['sin_freq']

            for k in self.controlled_dof:
                for _ in range(num_sin):
                    self.u_profile[:, k] += np.sin(2 * np.pi * sin_freq * np.random.rand() * sys_time)

        for k in self.controlled_dof:
            #self.u_profile[:, k] = np.clip(self.u_profile[:, k], a_min=-self.u_max, a_max=self.u_max)
            self.u_profile[:, k] = signal.lfilter(b, a, self.u_profile[:, k])
            self.u_profile[:, k] = np.clip(self.u_profile[:, k], a_min=-self.u_max, a_max=self.u_max)
            resamp[:, k] = self.u_profile[::self.ctrl_rate, k]

        if plot_profile:
            plt.figure('Control profile')
            for k in range(self.num_dof):
                plt.plot(sys_time, self.u_profile[:, k], label=r'$u_{}$'.format(k+1))
                plt.step(sys_time[::self.ctrl_rate], resamp[:, k], label=r'$\hat{u}_'+str(k + 1)+'$', where='post')
            plt.xlabel('time [s]')
            plt.ylabel('torque [Nm]')

            plt.legend()
            plt.show()


    def get_control_output_(self, x, t):
        if self.ctrl_cnt % self.ctrl_rate == 0 and self.ctrl_cnt >= self.wait_steps:
            self.last_control = self.u_profile[int(t * self.system_freq)-self.wait_steps, :]

        self.ctrl_cnt += 1
        return self.last_control


#c = Controller_Random_exploration(10, 4, 0, controlled_dof=[1], random_par={'std': 10, 'butter_order': 2})
#c = Controller_Random_exploration(10, 5, 1, type_random='SUM_SIN', controlled_dof=[1], random_par = {'sin_freq': 5, 'num_sin': 5, 'butter_order': 2})