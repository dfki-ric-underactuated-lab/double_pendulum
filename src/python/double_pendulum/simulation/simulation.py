import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation

from double_pendulum.simulation.visualization import get_arrow, \
                                                     set_arrow_properties
from double_pendulum.experiments.filters.low_pass import lowpass_filter
from double_pendulum.experiments.filters.kalman_filter import kalman_filter_rt


class Simulator:
    def __init__(self, plant):
        self.plant = plant

        self.x = np.zeros(2*self.plant.dof)  # position, velocity
        self.t = 0.0  # time

    def set_state(self, time, x):
        self.x = np.copy(x)
        self.t = time

    def get_state(self):
        return self.t, self.x

    def reset_data_recorder(self):
        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def record_data(self, time, x, tau=None):
        self.t_values.append(time)
        self.x_values.append(list(x))
        if tau is not None:
            self.tau_values.append(list(tau))

    def get_trajectory_data(self):
        T = np.asarray(self.t_values)
        X = np.asarray(self.x_values)
        U = np.asarray(self.tau_values)
        return T, X, U

    def set_imperfections(self,
                          noise_amplitude=0.0,
                          noise_mode="None",
                          noise_cut=0.0,
                          noise_vfilter="lowpass",
                          noise_vfilter_args={"alpha": 0.3},
                          delay=0.0,
                          delay_mode="None",
                          unoise_amplitude=0.0,
                          u_responsiveness=1.0,
                          perturbation_times=[],
                          perturbation_taus=[]):

        self.noise_amplitude = noise_amplitude
        self.noise_mode = noise_mode
        self.noise_cut = noise_cut
        self.noise_vfilter = noise_vfilter
        self.noise_vfilter_args = noise_vfilter_args
        self.delay = delay
        self.delay_mode = delay_mode
        self.unoise_amplitude = unoise_amplitude
        self.u_responsiveness = u_responsiveness
        self.perturbation_times = perturbation_times
        self.perturbation_taus = perturbation_taus

        self.imp_x_values = []
        self.con_u_values = []

        if noise_vfilter == "kalman":
            noise = np.zeros((2*self.plant.dof, 2*self.plant.dof))
            if noise_mode == "posvel":
                noise[0, 0] = noise_amplitude
                noise[1, 1] = noise_amplitude
            noise[2, 2] = noise_amplitude
            noise[3, 3] = noise_amplitude

            self.kalman_filter = kalman_filter_rt(
                    dim_x=2*self.plant.dof,
                    dim_u=self.plant.n_actuators,
                    measurement_noise=noise)

    def reset_imperfections(self):

        self.noise_amplitude = 0.0
        self.noise_mode = "None"
        self.noise_cut = 0.0
        self.noise_vfilter = "lowpass"
        self.noise_vfilter_args = {"alpha": 0.3}
        self.delay = 0.0
        self.delay_mode = "None"
        self.unoise_amplitude = 0.0
        self.u_responsiveness = 1.0
        self.perturbation_times = []
        self.perturbation_taus = []

        self.imp_x_values = []
        self.con_u_values = []
        self.kalman_filter = None

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
        self.t += dt
        self.record_data(self.t, self.x.copy(), tau)

    def controller_step(self, dt, controller, integrator):
        realtime = True
        if controller is not None:
            t0 = time.time()
            tau = controller.get_control_output(x=self.x, t=self.t)
            if time.time() - t0 > dt:
                realtime = False
        else:
            tau = np.zeros(self.plant.n_actuators)
        self.step(tau, dt, integrator=integrator)
        return realtime

    def controller_step_with_imperfections(self,
                                           dt,
                                           controller=None,
                                           integrator="runge_kutta"):

        # delay
        n_delay = int(self.delay / dt) + 1
        xcon = np.copy(self.x)
        if n_delay > 0:
            len_X = len(self.x_values)
            if self.delay_mode == "posvel":
                xcon = np.copy(self.x_values[max(-n_delay, -len_X)])
            elif self.delay_mode == "vel":
                # xcon[:2] = self.x[:2]
                xcon[2:] = self.x_values[max(-n_delay, -len_X)][2:]

        # noise
        if self.noise_mode == "posvel":
            # add noise to full state
            xcon = xcon + np.random.uniform(-self.noise_amplitude,
                                            self.noise_amplitude,
                                            np.shape(self.x))
        elif self.noise_mode == "vel":
            xcon[2:] = xcon[2:] + np.random.uniform(-self.noise_amplitude,
                                                    self.noise_amplitude,
                                                    np.shape(self.x[2:]))
        elif self.noise_mode == "velcut":
            # add noise to vel and cut off small velocities
            xcon[2:] = xcon[2:] + np.random.uniform(-self.noise_amplitude,
                                                    self.noise_amplitude,
                                                    np.shape(self.x[2:]))
            xcon[2] = np.where(np.abs(xcon[2]) < self.noise_cut, 0, xcon[2])
            xcon[3] = np.where(np.abs(xcon[3]) < self.noise_cut, 0, xcon[3])
        elif self.noise_mode == "velfilt":
            xcon[2:] = xcon[2:] + np.random.uniform(-self.noise_amplitude,
                                                    self.noise_amplitude,
                                                    np.shape(self.x[2:]))
            if len(self.imp_x_values) > 0:
                vf1 = [self.imp_x_values[-1][2], xcon[2]]
                vf2 = [self.imp_x_values[-1][3], xcon[3]]

                if self.noise_vfilter == "lowpass":
                    xcon[2] = lowpass_filter(
                            vf1,
                            self.noise_vfilter_args["alpha"])[-1]
                    xcon[3] = lowpass_filter(
                            vf2,
                            self.noise_vfilter_args["alpha"])[-1]

                elif self.noise_vfilter == "kalman":
                    A, B = self.plant.linear_matrices(
                            self.imp_x_values[-1],
                            self.con_u_values[-1])
                    xcon = self.kalman_filter(A, B,
                                xcon,  # self.imp_x_values[-1] or xcon?
                                self.con_u_values[-1])

        elif self.noise_mode == "velcutfilt":
            xcon[2:] = xcon[2:] + np.random.uniform(-self.noise_amplitude,
                                                    self.noise_amplitude,
                                                    np.shape(self.x[2:]))
            xcon[2] = np.where(np.abs(xcon[2]) < self.noise_cut, 0, xcon[2])
            xcon[3] = np.where(np.abs(xcon[3]) < self.noise_cut, 0, xcon[3])
            if len(self.imp_x_values) > 0:
                vf1 = [self.imp_x_values[-1][2], xcon[2]]
                vf2 = [self.imp_x_values[-1][3], xcon[3]]

                if self.noise_vfilter == "lowpass":
                    xcon[2] = lowpass_filter(
                            vf1,
                            self.noise_vfilter_args["alpha"])[-1]
                    xcon[3] = lowpass_filter(
                            vf2,
                            self.noise_vfilter_args["alpha"])[-1]

            self.imp_x_filter.append(xcon)

        realtime = True
        if controller is not None:
            t0 = time.time()
            u = controller.get_control_output(x=xcon, t=self.t)
            if time.time() - t0 > dt:
                realtime = False
        else:
            u = np.zeros(self.plant.n_actuators)

        self.con_u_values.append(u)

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
                nu[i] = tau + np.random.uniform(-self.unoise_amplitude,
                                                self.unoise_amplitude,
                                                1)
        self.step(nu, dt, integrator=integrator)
        return realtime

    def simulate(self, t0, x0, tf, dt, controller=None,
                 integrator="runge_kutta", imperfections=False):
        self.set_state(t0, x0)
        self.reset_data_recorder()
        self.record_data(t0, np.copy(x0), None)

        if imperfections:
            if self.noise_vfilter == "kalman":
                self.kalman_filter.set_parameters(x0=x0, dt=dt)
                self.kalman_filter.init()

        while (self.t <= tf):
            # if controller is not None:
            #     tau = controller.get_control_output(x=self.x, t=self.t)
            # else:
            #     tau = np.zeros(self.plant.n_actuators)
            # self.step(tau, dt, integrator=integrator)
            if not imperfections:
                _ = self.controller_step(dt, controller, integrator)
            else:
                _ = self.controller_step_with_imperfections(
                        dt, controller, integrator)

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

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        x0 = par_dict["x0"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        imperfections = par_dict["imperfections"]
        anim_dt = par_dict["anim_dt"]
        trail_len = 25  # length of the trails
        sim_steps = int(anim_dt / dt)

        if imperfections:
            if self.noise_vfilter == "kalman":
                self.kalman_filter.set_parameters(x0=x0, dt=dt)
                self.kalman_filter.init()

        realtime = True
        for _ in range(sim_steps):
            # if controller is not None:
            #     t0 = time.time()
            #     tau = controller.get_control_output(x=self.x, t=self.t)
            #     if time.time() - t0 > dt:
            #         realtime = False
            # else:
            #     tau = np.zeros(self.plant.n_actuators)
            # self.step(tau, dt, integrator=integrator)
            if not imperfections:
                rt = self.controller_step(dt, controller, integrator)
            else:
                rt = self.controller_step_with_imperfections(
                        dt, controller, integrator)
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
                             integrator="runge_kutta", imperfections=False,
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
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["x0"] = x0
        par_dict["anim_dt"] = anim_dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        par_dict["imperfections"] = imperfections
        frames = num_steps*[par_dict]

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
