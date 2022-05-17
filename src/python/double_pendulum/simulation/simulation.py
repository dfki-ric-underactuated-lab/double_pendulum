import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation

from double_pendulum.simulation.visualization import get_arrow, \
                                                     set_arrow_properties
from double_pendulum.utils.filters.low_pass import lowpass_filter_rt
from double_pendulum.utils.filters.kalman_filter import kalman_filter_rt


class Simulator:
    def __init__(self, plant):
        self.plant = plant

        self.x = np.zeros(2*self.plant.dof)  # position, velocity
        self.t = 0.0  # time

        self.reset()

    def set_state(self, t, x):
        self.x = np.copy(x)
        self.t = t

    def get_state(self):
        return self.t, self.x

    def reset_data_recorder(self):
        self.t_values = []
        self.x_values = []
        self.tau_values = []

        self.meas_x_values = []
        self.filt_x_values = []
        self.con_u_values = []

    def record_data(self, t, x, tau=None):
        self.t_values.append(t)
        self.x_values.append(list(x))
        if tau is not None:
            self.tau_values.append(list(tau))

    def get_trajectory_data(self):
        T = np.asarray(self.t_values)
        X = np.asarray(self.x_values)
        U = np.asarray(self.tau_values)
        return T, X, U

    def set_process_noise(self,
                          process_noise_sigmas=[0., 0., 0., 0.]):
        self.process_noise_sigmas = process_noise_sigmas

    def set_measurement_parameters(self,
                                   C=np.eye(4),
                                   D=np.zeros((4,2)),
                                   #noise_mode="None",
                                   #noise_amplitude=0.0,
                                   x_noise_sigmas=[0., 0., 0., 0.],
                                   delay=0.0,
                                   delay_mode="None"):

        self.meas_C = C
        self.meas_D = D
        #self.noise_amplitude = noise_amplitude
        #self.noise_mode = noise_mode
        self.x_noise_sigmas = x_noise_sigmas
        self.delay = delay
        self.delay_mode = delay_mode

    def set_filter_parameters(self,
                              noise_cut=0.0,
                              noise_vfilter="lowpass",
                              noise_vfilter_args={"alpha": [1., 1., 1.0, 1.0]}):

        self.noise_cut = noise_cut
        self.noise_vfilter = noise_vfilter
        self.noise_vfilter_args = noise_vfilter_args

    def set_motor_parameters(self,
                             u_noise_amplitude=0.0,
                             u_responsiveness=1.0):
        self.unoise_amplitude = u_noise_amplitude
        self.u_responsiveness = u_responsiveness

    def set_disturbances(self,
                         perturbation_times=[],
                         perturbation_taus=[]):

        self.perturbation_times = perturbation_times
        self.perturbation_taus = perturbation_taus

    def reset(self):

        self.process_noise_sigmas = [0., 0., 0., 0.]
        #self.noise_amplitude = 0.0
        #self.noise_mode = "None"
        self.x_noise_sigmas = [0., 0., 0., 0.]
        self.delay = 0.0
        self.delay_mode = "None"
        self.noise_cut = 0.0
        self.noise_vfilter = "lowpass"
        self.noise_vfilter_args = {"alpha": [1., 1., 1.0, 1.0]}
        self.unoise_amplitude = 0.0
        self.u_responsiveness = 1.0
        self.perturbation_times = []
        self.perturbation_taus = []
        self.filter = None

        self.reset_data_recorder()

    def init_filter(self, x0, dt):
        if self.noise_vfilter == "lowpass":
            dof = self.plant.dof

            self.filter = lowpass_filter_rt(
                    dim_x=2*dof,
                    alpha=self.noise_vfilter_args["alpha"],
                    x0=x0)

        if self.noise_vfilter == "kalman":
            dof = self.plant.dof
            noise = np.zeros((2*dof, 2*dof))  # not used
            # for i in range(dof):
            #     noise[dof+i, dof+i] = self.x_noise_sigmas

            self.filter = kalman_filter_rt(
                    plant=self.plant,
                    dim_x=2*dof,
                    dim_u=self.plant.n_actuators,
                    x0=x0,
                    dt=dt,
                    measurement_noise=noise)

    def euler_integrator(self, t, y, dt, tau):
        return self.plant.rhs(t, y, tau)

    def runge_integrator(self, t, y, dt, tau):
        k1 = self.plant.rhs(t, y, tau)
        k2 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, tau)
        k3 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, tau)
        k4 = self.plant.rhs(t + dt, y + dt * k3, tau)
        return (k1 + 2 * (k2 + k3) + k4) / 6.0

    def step(self, tau, dt, integrator="runge_kutta"):
        tau = np.clip(tau, -np.asarray(self.plant.torque_limit),
                      np.asarray(self.plant.torque_limit))

        # self.record_data(self.t, self.x.copy(), tau)

        if integrator == "runge_kutta":
            self.x += dt * self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x += dt * self.euler_integrator(self.t, self.x, dt, tau)
        else:
            raise NotImplementedError(
                   f'Sorry, the integrator {integrator} is not implemented.')
        # process noise
        self.x = np.random.normal(self.x, self.process_noise_sigmas, np.shape(self.x))

        self.t += dt
        self.record_data(self.t, self.x.copy(), tau)

    def get_control_u(self, controller, x, t, dt):
        realtime = True
        if controller is not None:
            t0 = time.time()
            u = controller.get_control_output(x=x, t=t)
            if time.time() - t0 > dt:
                realtime = False
        else:
            u = np.zeros(self.plant.n_actuators)
        self.con_u_values.append(np.copy(u))
        return u, realtime

    def get_measurement(self, dt):

        x_meas = np.copy(self.x)

        # delay
        n_delay = int(self.delay / dt) + 1
        if n_delay > 1:
            len_X = len(self.x_values)
            if self.delay_mode == "posvel":
                x_meas = np.copy(self.x_values[max(-n_delay, -len_X)])
            elif self.delay_mode == "vel":
                # x_meas[:2] = self.x[:2]
                x_meas[2:] = self.x_values[max(-n_delay, -len_X)][2:]

        if len(self.tau_values) > n_delay:
            u = np.asarray(self.tau_values[-n_delay])
        else:
            u = np.zeros(self.plant.n_actuators)

        x_meas = np.dot(self.meas_C, x_meas) + np.dot(self.meas_D, u)

        # sensor noise
        x_meas = np.random.normal(x_meas, self.x_noise_sigmas, np.shape(self.x))

        self.meas_x_values.append(np.copy(x_meas))
        return x_meas

    def filter_measurement(self, x):
        x_filt = np.copy(x)

        # velocity cut
        if self.noise_cut > 0.:
            x_filt[2] = np.where(np.abs(x_filt[2]) < self.noise_cut, 0, x_filt[2])
            x_filt[3] = np.where(np.abs(x_filt[3]) < self.noise_cut, 0, x_filt[3])

        # filter
        if not self.filter is None:
            if len(self.con_u_values) > 0:
                x_filt = self.filter(x, self.con_u_values[-1])

        self.filt_x_values.append(np.copy(x_filt))
        return x_filt

    def get_real_applied_u(self, u):
        nu = np.copy(u)

        # tau responsiveness
        if len(self.tau_values) > 0:
            last_u = np.asarray(self.tau_values[-1])
        else:
            last_u = np.zeros(self.plant.n_actuators)
        nu = last_u + self.u_responsiveness*(nu - last_u)

        # tau noise (unoise)
        for i, tau in enumerate(nu):
            if np.abs(tau) > 0:
                # nu[i] = tau + np.random.uniform(-self.unoise_amplitude,
                #                                 self.unoise_amplitude,
                #                                 1)
                nu[i] = tau + np.random.normal(0,
                                               self.unoise_amplitude,
                                               1)
        return nu

    def controller_step(self,
                        dt,
                        controller=None,
                        integrator="runge_kutta"):

        x_meas = self.get_measurement(dt)
        x_filt = self.filter_measurement(x_meas)

        u, realtime = self.get_control_u(controller, x_filt, self.t, dt)
        nu = self.get_real_applied_u(u)

        self.step(nu, dt, integrator=integrator)

        return realtime

    def simulate(self, t0, x0, tf, dt, controller=None,
            integrator="runge_kutta"): #, imperfections=False):
        self.set_state(t0, x0)
        self.reset_data_recorder()
        self.record_data(t0, np.copy(x0), None)

        if not self.filter is None:
            self.init_filter(x0, dt)

        while (self.t <= tf):
            _ = self.controller_step(dt, controller, integrator)

        return self.t_values, self.x_values, self.tau_values

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(self.plant.workspace_range[0][0],
                                   self.plant.workspace_range[0][1])
        self.animation_ax.set_ylim(self.plant.workspace_range[1][0],
                                   self.plant.workspace_range[1][1])
        self.animation_ax.get_xaxis().set_visible(False)
        self.animation_ax.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.tight_layout()
        for ap in self.animation_plots[:-1]:
            ap.set_data([], [])
        self.animation_plots[-1].set_text("t = 0.000")

        self.ee_poses = []
        self.tau_arrowarcs = []
        self.tau_arrowheads = []
        for link in range(self.plant.n_links):
            arc, head = get_arrow(radius=0.001,
                                  centX=0,
                                  centY=0,
                                  angle_=110,
                                  theta2_=320,
                                  color_="red")
            self.tau_arrowarcs.append(arc)
            self.tau_arrowheads.append(head)
            self.animation_ax.add_patch(arc)
            self.animation_ax.add_patch(head)

        dt = self.par_dict["dt"]
        x0 = self.par_dict["x0"]
        # imperfections = self.par_dict["imperfections"]
        # if imperfections:
        if not self.filter is None:
            self.init_filter(x0, dt)

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        # x0 = par_dict["x0"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        # imperfections = par_dict["imperfections"]
        anim_dt = par_dict["anim_dt"]
        trail_len = 25  # length of the trails
        sim_steps = int(anim_dt / dt)

        realtime = True
        for _ in range(sim_steps):
            rt = self.controller_step(dt, controller, integrator)
            if not rt:
                realtime = False
        tau = self.tau_values[-1]
        ee_pos = self.plant.forward_kinematics(self.x[:self.plant.dof])
        ee_pos.insert(0, self.plant.base)

        self.ee_poses.append(ee_pos)
        if len(self.ee_poses) > trail_len:
            self.ee_poses = np.delete(self.ee_poses, 0, 0).tolist()

        ani_plot_counter = 0
        # plot links
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(
                            [ee_pos[link][0], ee_pos[link+1][0]],
                            [ee_pos[link][1], ee_pos[link+1][1]])
            ani_plot_counter += 1

        # plot base
        self.animation_plots[ani_plot_counter].set_data(ee_pos[0][0],
                                                        ee_pos[0][1])
        ani_plot_counter += 1

        # plot bodies
        for link in range(self.plant.n_links):

            self.animation_plots[ani_plot_counter].set_data(ee_pos[link+1][0],
                                                            ee_pos[link+1][1])
            ani_plot_counter += 1

            if self.plot_trail:
                self.animation_plots[ani_plot_counter].set_data(
                        np.asarray(self.ee_poses)[:, link+1, 0],
                        np.asarray(self.ee_poses)[:, link+1, 1])
                ani_plot_counter += 1

            set_arrow_properties(self.tau_arrowarcs[link],
                                 self.tau_arrowheads[link],
                                 tau[link],
                                 ee_pos[link][0],
                                 ee_pos[link][1])

        if self.plot_inittraj:
            T, X, U = controller.get_init_trajectory()
            coords = []
            for x in X:
                coords.append(
                    self.plant.forward_kinematics(x[:self.plant.dof])[-1])

            coords = np.asarray(coords)
            self.animation_plots[ani_plot_counter].set_data(coords.T[0],
                                                            coords.T[1])
            ani_plot_counter += 1

        if self.plot_forecast:
            T, X, U = controller.get_forecast()
            coords = []
            for x in X:
                coords.append(
                    self.plant.forward_kinematics(x[:self.plant.dof])[-1])

            coords = np.asarray(coords)
            self.animation_plots[ani_plot_counter].set_data(coords.T[0],
                                                            coords.T[1])
            ani_plot_counter += 1

        t = float(self.animation_plots[ani_plot_counter].get_text()[4:])
        t = round(t+dt*sim_steps, 3)
        self.animation_plots[ani_plot_counter].set_text(f"t = {t}")

        # if the animation runs slower than real time
        # the time display will be red
        if (not realtime):
            self.animation_plots[ani_plot_counter].set_color("red")
        else:
            self.animation_plots[ani_plot_counter].set_color("black")

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def simulate_and_animate(self, t0, x0, tf, dt, controller=None,
                             integrator="runge_kutta",# imperfections=False,
                             plot_inittraj=False, plot_forecast=False,
                             plot_trail=True,
                             phase_plot=False, save_video=False,
                             video_name="pendulum_swingup", anim_dt=0.02):
        """
        Simulation and animation of the pendulum motion
        The animation is only implemented for 2d serial chains
        """

        self.plot_inittraj = plot_inittraj
        self.plot_forecast = plot_forecast
        self.plot_trail = plot_trail
        self.set_state(t0, x0)
        self.reset_data_recorder()
        self.record_data(t0, np.copy(x0), None)

        fig = plt.figure(figsize=(20, 20))
        self.animation_ax = plt.axes()
        self.animation_plots = []

        colors = ['#0077BE', '#f66338']
        colors_trails = ['#d2eeff', '#ffebd8']

        for link in range(self.plant.n_links):
            bar_plot, = self.animation_ax.plot([], [], "-",
                                               lw=10, color='k')
            self.animation_plots.append(bar_plot)

        base_plot, = self.animation_ax.plot([], [], "s",
                                            markersize=25.0, color="black")
        self.animation_plots.append(base_plot)
        for link in range(self.plant.n_links):
            ee_plot, = self.animation_ax.plot(
                    [], [], "o",
                    markersize=50.0,
                    color='k',
                    markerfacecolor=colors[link % len(colors)])
            self.animation_plots.append(ee_plot)
            if self.plot_trail:
                trail_plot, = self.animation_ax.plot(
                        [], [], '-',
                        color=colors[link],
                        markersize=24,
                        markerfacecolor=colors_trails[link % len(colors_trails)],
                        lw=2,
                        markevery=10000,
                        markeredgecolor='None')
                self.animation_plots.append(trail_plot)

        if self.plot_inittraj:
            it_plot, = self.animation_ax.plot([], [], "--",
                                              lw=1, color="gray")
            self.animation_plots.append(it_plot)
        if self.plot_forecast:
            fc_plot, = self.animation_ax.plot([], [], "-",
                                              lw=1, color="green")
            self.animation_plots.append(fc_plot)

        text_plot = self.animation_ax.text(0.1, 0.9, [],
                                           fontsize=60,
                                           transform=fig.transFigure)

        self.animation_plots.append(text_plot)

        num_steps = int(tf / anim_dt)
        self.par_dict = {}
        self.par_dict["dt"] = dt
        self.par_dict["x0"] = x0
        self.par_dict["anim_dt"] = anim_dt
        self.par_dict["controller"] = controller
        self.par_dict["integrator"] = integrator
        # self.par_dict["imperfections"] = imperfections
        frames = num_steps*[self.par_dict]

        animation = FuncAnimation(fig, self._animation_step, frames=frames,
                                  init_func=self._animation_init, blit=True,
                                  repeat=False, interval=dt*1000)

        if save_video:
            print(f"Saving video to {video_name}.mp4")
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            animation.save(video_name+'.mp4', writer=writer)
            print("Saving video done.")
        else:
            self.set_state(t0, x0)
            self.reset_data_recorder()
            self.record_data(t0, np.copy(x0), None)
            plt.show()
        plt.close()

        return self.t_values, self.x_values, self.tau_values
