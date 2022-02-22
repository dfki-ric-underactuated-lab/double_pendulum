import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
import time

from double_pendulum.simulation.visualization import get_arrow, \
                                                     set_arrow_properties


class Simulator:
    def __init__(self, plant):
        self.plant = plant

        self.x = np.zeros(2*self.plant.dof)  # position, velocity
        self.t = 0.0  # time

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def reset_data_recorder(self):
        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def record_data(self, time, x, tau):
        self.t_values.append(time)
        self.x_values.append(x)
        self.tau_values.append(tau)

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

        self.record_data(self.t, self.x.copy(), tau)

        if integrator == "runge_kutta":
            self.x += dt * self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x += dt * self.euler_integrator(self.t, self.x, dt, tau)
        else:
            raise NotImplementedError(
                   f'Sorry, the integrator {integrator} is not implemented.')
        self.t += dt
        # self.record_data(self.t, self.x.copy(), tau)

    def simulate(self, t0, x0, tf, dt, controller=None,
                 integrator="runge_kutta"):
        self.set_state(t0, x0)
        self.reset_data_recorder()

        while (self.t <= tf):
            if controller is not None:
                tau = controller.get_control_output(x=self.x, t=self.t)
            else:
                tau = np.zeros(self.plant.n_actuators)
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(self.plant.workspace_range[0][0],
                                   self.plant.workspace_range[0][1])
        self.animation_ax.set_ylim(self.plant.workspace_range[1][0],
                                   self.plant.workspace_range[1][1])
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots[:-1]:
            ap.set_data([], [])
        self.animation_plots[-1].set_text("t = 0.000")

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
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        realtime = True
        if controller is not None:
            t0 = time.time()
            tau = controller.get_control_output(x=self.x, t=self.t)
            if time.time() - t0 > dt:
                realtime = False
        else:
            tau = np.zeros(self.plant.n_actuators)
        self.step(tau, dt, integrator=integrator)
        ee_pos = self.plant.forward_kinematics(self.x[:self.plant.dof])
        ee_pos.insert(0, self.plant.base)
        ani_plot_counter = 0
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(ee_pos[link+1][0],
                                                            ee_pos[link+1][1])
            ani_plot_counter += 1

            self.animation_plots[ani_plot_counter].set_data(
                            [ee_pos[link][0], ee_pos[link+1][0]],
                            [ee_pos[link][1], ee_pos[link+1][1]])
            ani_plot_counter += 1

            set_arrow_properties(self.tau_arrowarcs[link],
                                 self.tau_arrowheads[link],
                                 tau[link],
                                 ee_pos[link][0],
                                 ee_pos[link][1])

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
        t = round(t+dt, 3)
        self.animation_plots[ani_plot_counter].set_text(f"t = {t}")

        # if the animation runs slower than real time
        # the time display will be red
        if (not realtime):
            self.animation_plots[ani_plot_counter].set_color("red")
        else:
            self.animation_plots[ani_plot_counter].set_color("black")

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _ps_init(self):
        """
        init of the phase space animation plot
        """
        self.ps_ax.set_xlim(-np.pi, np.pi)
        self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_xlabel("degree [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.ps_plots:
            ap.set_data([], [])
        return self.ps_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """
        for d in range(self.plant.dof):
            self.ps_plots[d].set_data(
                            np.asarray(self.x_values).T[d],
                            np.asarray(self.x_values).T[self.plant.dof+d])
        return self.ps_plots

    def simulate_and_animate(self, t0, x0, tf, dt, controller=None,
                             integrator="runge_kutta", plot_forecast=False,
                             phase_plot=False, save_video=False,
                             video_name="pendulum_swingup"):
        """
        Simulation and animation of the pendulum motion
        The animation is only implemented for 2d serial chains
        """

        self.plot_forecast = plot_forecast
        self.set_state(t0, x0)
        self.reset_data_recorder()

        fig = plt.figure(figsize=(20, 20))
        self.animation_ax = plt.axes()
        self.animation_plots = []
        if self.plot_forecast:
            self.forecast_plots = []

        for link in range(self.plant.n_links):
            ee_plot, = self.animation_ax.plot([], [], "o",
                                              markersize=25.0, color="blue")
            self.animation_plots.append(ee_plot)

            bar_plot, = self.animation_ax.plot([], [], "-",
                                               lw=5, color="black")
            self.animation_plots.append(bar_plot)

        if self.plot_forecast:
            fc_plot, = self.animation_ax.plot([], [], "--",
                                              lw=1, color="gray")
            self.animation_plots.append(fc_plot)

        text_plot = self.animation_ax.text(0.15, 0.85, [],
                                           fontsize=40,
                                           transform=fig.transFigure)

        self.animation_plots.append(text_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]

        animation = FuncAnimation(fig, self._animation_step, frames=frames,
                                  init_func=self._animation_init, blit=True,
                                  repeat=False, interval=dt*1000)

        if phase_plot:
            ps_fig = plt.figure(figsize=(10, 10))
            self.ps_ax = plt.axes()
            self.ps_plots = []
            for d in range(self.plant.dof):
                ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
                self.ps_plots.append(ps_plot)

            animation2 = FuncAnimation(ps_fig, self._ps_update,
                                       init_func=self._ps_init, blit=True,
                                       repeat=False, interval=dt*1000)

        if save_video:
            print(f"Saving video to {video_name}.mp4")
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            animation.save(video_name+'.mp4', writer=writer)
            # if phase_plot:
            #     self.set_state(t0, x0)
            #     self.reset_data_recorder()
            #     Writer2 = mplanimation.writers['ffmpeg']
            #     writer2 = Writer2(fps=60, bitrate=1800)
            #     animation2.save(video_name+'_phase.mp4', writer=writer2)
            print("Saving video done.")
        self.set_state(t0, x0)
        self.reset_data_recorder()
        plt.show()

        return self.t_values, self.x_values, self.tau_values
