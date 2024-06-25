import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation

from double_pendulum.simulation.visualization import get_arrow, set_arrow_properties


class Simulator:
    """
    Simulator class
    simulates and optionally animates the double pendulum motion.
    Animation is done with matplotlib Funcanimation.

    Parameters
    ----------
    plant : SymbolicDoublePendulum or DoublePendulumPlant object
        A plant object containing the kinematics and dynamics of the
        double pendulum
    """

    def __init__(self, plant):
        self.plant = plant

        self.x = np.zeros(2 * self.plant.dof)  # position, velocity
        self.t = 0.0  # time

        self.reset()

    def set_state(self, t, x):
        """
        Set the time and state of the double pendulum

        Parameters
        ----------
        t : float
            time, units=[s]
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        self.x = np.copy(x)
        self.t = t

    def get_state(self):
        """
        Get the double pendulum state

        Returns
        -------
        float
            time, unit=[s]
        numpy_array
            shape=(4,)
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        return self.t, self.x

    def reset_data_recorder(self):
        """
        Reset the internal data record of the simulator
        """
        self.t_values = []
        self.x_values = []
        self.tau_values = []

        self.meas_x_values = []
        self.con_u_values = []

    def record_data(self, t, x, tau=None):
        """
        Record a data point in the simulator's internal record

        Parameters
        ----------
        t : float
            time, units=[s]

        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]

        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        """
        self.t_values.append(t)
        self.x_values.append(list(x))
        if tau is not None:
            self.tau_values.append(list(tau))

    def get_trajectory_data(self):
        """
        Get the rocrded trajectory data

        Returns
        -------
        numpy_array
            time points, unit=[s]
            shape=(N,)
        numpy_array
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        numpy_array
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """
        T = np.asarray(self.t_values)
        X = np.asarray(self.x_values)
        U = np.asarray(self.tau_values)
        return T, X, U

    def set_process_noise(self, process_noise_sigmas=[0.0, 0.0, 0.0, 0.0]):
        """
        Set parameters for process noise (Gaussian)

        Parameters
        ----------
        process_noise_sigmas : array_like
            shape=(4,)
            Gaussian standard deviations for the process noise.
            Each entry in the list corresponds to a state variable.
            (Default value = [0., 0., 0., 0.])
        """
        self.process_noise_sigmas = process_noise_sigmas

    def set_measurement_parameters(
        self,
        C=np.eye(4),
        D=np.zeros((4, 2)),
        meas_noise_sigmas=[0.0, 0.0, 0.0, 0.0],
        delay=0.0,
        delay_mode="None",
    ):
        """
        Set parameters for state measuremts

        The state measurement is described by
        x_meas(t) = C*x(t-delay) + D*u(t-delay) + N(sigma)

        Parameters
        ----------
        C : numpy_array
            state-state measurement matrix
            (Default value = np.eye(4))
        D : numpy_array
            state-torque measurement matrix
            (Default value = np.zeros((4, 2))

        meas_noise_sigmas : array_like
            Standard deviations of Gaussian measurement noise
            (Default value = [0., 0., 0., 0.])
        delay : float
            time delay of measurements, unit=[s]
             (Default value = 0.0)
        delay_mode : string
            string determining what state variables are delayed:
            "None": no delay
            "vel": velocity measurements are delayed
            "posvel": position and velocity measurements are delayed
             (Default value = "None")
        """
        self.meas_C = C
        self.meas_D = D
        self.meas_noise_sigmas = meas_noise_sigmas
        self.delay = delay
        self.delay_mode = delay_mode

    def set_motor_parameters(self, u_noise_sigmas=[0.0, 0.0], u_responsiveness=1.0):
        """
        Set parameters for the motors

        The applied motor torque (u_out) is related to the commanded torque
        (u) and the last torque output (u_last) via

        u_out = u_responsiveness*u + (1-u_responsiveness)*u_last + N(sigma)

        Parameters
        ----------
        u_noise_sigmas : array_like
            shape=(2,)
            Standard deviation of the gaussian noise for the torque produced by
            the motors
            (Default value = [0., 0.])
        u_responsiveness : float
            resonsiveness of the motors
            (Default value = 1.)
        """
        self.u_noise_sigmas = u_noise_sigmas
        self.u_responsiveness = u_responsiveness

    def set_disturbances(self, perturbation_array=[[], []]):
        """
        Set disturbances (hits) happening during the simulation.
        (Not yet implemented)

        Parameters
        ----------
        perturbation_array : array_like
             (Default value = [[], []])
             List of two lists.
             First list: Perturbations on first joint,
             Second list: Perturbations on second joint
             The lists should contain the torque pertubations for the two
             joints for every timestep.
        """
        self.perturbation_array = perturbation_array

    def reset(self):
        """
        Reset the Simulator
        Resets
            - the internal data recorder
            - the filter + arguments
            - the process noise
            - the measurement parameters
            - the motor parameters
            - perturbations
        """

        self.process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]

        self.meas_C = np.eye(4)
        self.meas_D = np.zeros((4, 2))
        self.meas_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
        self.delay = 0.0
        self.delay_mode = "None"

        self.u_noise_sigmas = [0.0, 0.0]
        self.u_responsiveness = 1.0

        self.perturbation_array = [[], []]

        self.reset_data_recorder()

    def euler_integrator(self, y, dt, t, tau):
        """
        Performs a Euler integration step

        Parameters
        ----------
        y : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        dt : float
            timestep, unit=[s]
        t : float
            time, unit=[s]
        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        numpy_array
            shape=(4,), dtype=float,
            new state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        return self.plant.rhs(t, y, tau)

    def runge_integrator(self, y, dt, t, tau):
        """
        Performs a Runge-Kutta integration step

        Parameters
        ----------
        y : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        dt : float
            timestep, unit=[s]
        t : float
            time, unit=[s]
        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        Returns
        -------
        numpy_array
            shape=(4,), dtype=float,
            new state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """
        k1 = self.plant.rhs(t, y, tau)
        k2 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, tau)
        k3 = self.plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, tau)
        k4 = self.plant.rhs(t + dt, y + dt * k3, tau)
        return (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    def step(self, tau, dt, integrator="runge_kutta"):
        """
        Performs a simulation step with the specified integrator.
        Also adds process noise to the integration result.
        Uses and updates the internal state

        Parameters
        ----------
        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        dt : float
            timestep, unit=[s]
        integrator : string
            string determining the integration method
            "euler" : Euler integrator
            "runge_kutta" : Runge Kutta integrator
             (Default value = "runge_kutta")
        """
        # tau = np.clip(
        #     tau,
        #     -np.asarray(self.plant.torque_limit),
        #     np.asarray(self.plant.torque_limit),
        # )

        if integrator == "runge_kutta":
            self.x = np.add(
                self.x,
                dt * self.runge_integrator(self.x, dt, self.t, tau),
                casting="unsafe",
            )
            # self.x += dt * self.runge_integrator(self.x, dt, self.t, tau)
        elif integrator == "euler":
            self.x = np.add(
                self.x,
                dt * self.euler_integrator(self.x, dt, self.t, tau),
                casting="unsafe",
            )
            # self.x += dt * self.euler_integrator(self.x, dt, self.t, tau)
        else:
            raise NotImplementedError(
                f"Sorry, the integrator {integrator} is not implemented."
            )
        # process noise
        self.x = np.random.normal(self.x, self.process_noise_sigmas, np.shape(self.x))

        self.t += dt
        self.record_data(self.t, self.x.copy(), tau)
        # _ = self.get_measurement(dt)

    def get_control_u(self, controller, x, t, dt):
        """
        Get the control signal from the controller

        Parameters
        ----------
        controller : Controller object
            Controller whose control signal is used
            If None, motir torques are set to 0.
        x : array_like, shape=(4,), dtype=float,
            state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        t : float,
            time, units=[s], not used
        dt : float
            timestep, unit=[s]

        Returns
        -------
        numpy_array
            shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        bool
            Flag stating real time calculation
            True: The calculation was performed in real time
            False: The calculation was not performed in real time
        """
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
        """
        Get a measurement from the internal state

        The state measurement is described by
        x_meas(t) = C*x(t-delay) + D*u(t-delay) + N(sigma)

        (parameters set by set_measurement_parameters)

        Parameters
        ----------
        dt : float
            timestep, unit=[s]

        Returns
        -------
        numpy_array
            shape=(4,), dtype=float,
            measured state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        """

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
        x_meas = np.random.normal(x_meas, self.meas_noise_sigmas, np.shape(self.x))

        self.meas_x_values.append(np.copy(x_meas))
        return x_meas

    def get_real_applied_u(self, u, t, dt):
        """
        Get the torque that the motor actually applies.

        The applied motor torque (u_out) is related to the commanded torque
        (u) and the last torque output (u_last) via

        u_out = u_responsiveness*u + (1-u_responsiveness)*u_last + N(sigma)

        (parameters set in set_motor_parameters)

        Parameters
        ----------
        tau : array_like, shape=(2,), dtype=float
            desired actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        t : float,
            start time, units=[s]
        dt : float
            timestep, unit=[s]

        Returns
        -------
        array-like
            shape=(2,), dtype=float
            actual actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]

        """
        nu = np.copy(u)

        # tau responsiveness
        if len(self.tau_values) > 0:
            last_u = np.asarray(self.tau_values[-1])
        else:
            last_u = np.zeros(self.plant.n_actuators)
        nu = last_u + self.u_responsiveness * (nu - last_u)

        # tau noise (unoise)
        nu = np.random.normal(nu, self.u_noise_sigmas, np.shape(nu))

        nu[0] = np.clip(nu[0], -self.plant.torque_limit[0], self.plant.torque_limit[0])
        nu[1] = np.clip(nu[1], -self.plant.torque_limit[1], self.plant.torque_limit[1])
        # perturbance
        # (can exceed joint limits)
        pert_index = int(t / dt)
        if pert_index < len(self.perturbation_array[0]):
            nu[0] += self.perturbation_array[0][pert_index]
        if pert_index < len(self.perturbation_array[1]):
            nu[1] += self.perturbation_array[1][pert_index]

        return nu

    def controller_step(self, dt, controller=None, integrator="runge_kutta"):
        """
        Perform a full simulation step including
            - get measurement
            - get controller signal
            - calculate actual applied torques
            - integrate the eom

        Parameters
        ----------
        dt : float
            timestep, unit=[s]
        controller : Controller object
            Controller whose control signal is used
            If None, motir torques are set to 0.
             (Default value = None)
        integrator : string
            string determining the integration method
            "euler" : Euler integrator
            "runge_kutta" : Runge Kutta integrator
             (Default value = "runge_kutta")

        Returns
        -------
        bool
            Flag stating real time calculation
            True: The calculation was performed in real time
            False: The calculation was not performed in real time
        """

        x_meas = self.get_measurement(dt)
        u, realtime = self.get_control_u(controller, x_meas, self.t, dt)
        nu = self.get_real_applied_u(u, self.t, dt)

        self.step(nu, dt, integrator=integrator)

        return realtime

    def simulate(self, t0, x0, tf, dt, controller=None, integrator="runge_kutta"):
        """
        Simulate the double pendulum for a time period under the control of a
        controller

        Parameters
        ----------
        t0 : float,
            start time, units=[s]
        x0 : array_like, shape=(4,), dtype=float,
            initial state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        tf : float
            final time, units=[s]
        dt : float
            timestep, unit=[s]
        controller : Controller object
            Controller whose control signal is used
            If None, motir torques are set to 0.
             (Default value = None)
        integrator : string
            string determining the integration method
            "euler" : Euler integrator
            "runge_kutta" : Runge Kutta integrator
             (Default value = "runge_kutta")

        Returns
        -------
        list
            time points, unit=[s]
            shape=(N,)
        list
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        list
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """
        self.set_state(t0, x0)
        self.reset_data_recorder()
        self.record_data(t0, np.copy(x0), None)
        # self.meas_x_values.append(np.copy(x0))

        # self.init_filter(x0, dt, integrator)

        N = 0
        while self.t < tf:
            _ = self.controller_step(dt, controller, integrator)
            N += 1

        return self.t_values, self.x_values, self.tau_values

    def _animation_init(self):
        """init of the animation plot"""
        self.animation_ax.set_xlim(
            self.plant.workspace_range[0][0], self.plant.workspace_range[0][1]
        )
        self.animation_ax.set_ylim(
            self.plant.workspace_range[1][0], self.plant.workspace_range[1][1]
        )
        self.animation_ax.get_xaxis().set_visible(False)
        self.animation_ax.get_yaxis().set_visible(False)
        plt.axis("off")
        plt.tight_layout()
        for ap in self.animation_plots[:-1]:
            ap.set_data([], [])
        t0 = self.par_dict["t0"]
        self.animation_plots[-1].set_text("t = " + str(round(t0, 3)))

        self.ee_poses = []
        self.tau_arrowarcs = []
        self.tau_arrowheads = []
        for link in range(self.plant.n_links):
            arc, head = get_arrow(
                radius=0.001, centX=0, centY=0, angle_=110, theta2_=320, color_="red"
            )
            self.tau_arrowarcs.append(arc)
            self.tau_arrowheads.append(head)
            self.animation_ax.add_patch(arc)
            self.animation_ax.add_patch(head)

        if self.plot_perturbations:
            for link in range(self.plant.n_links):
                arc, head = get_arrow(
                    radius=0.001,
                    centX=0,
                    centY=0,
                    angle_=110,
                    theta2_=320,
                    color_="purple",
                )
                self.tau_arrowarcs.append(arc)
                self.tau_arrowheads.append(head)
                self.animation_ax.add_patch(arc)
                self.animation_ax.add_patch(head)

        dt = self.par_dict["dt"]
        x0 = self.par_dict["x0"]
        integrator = self.par_dict["integrator"]
        # imperfections = self.par_dict["imperfections"]
        # if imperfections:
        # self.init_filter(x0, dt, integrator)

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def _animation_step(self, par_dict):
        """simulation of a single step which also updates the animation plot"""
        dt = par_dict["dt"]
        t0 = par_dict["t0"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        anim_dt = par_dict["anim_dt"]
        trail_len = 25  # length of the trails
        sim_steps = int(anim_dt / dt)
        dt_index = int(self.t / dt)

        realtime = True
        for _ in range(sim_steps):
            rt = self.controller_step(dt, controller, integrator)
            if not rt:
                realtime = False
        # tau = self.tau_values[-1]
        tau = self.con_u_values[-1]
        ee_pos = self.plant.forward_kinematics(self.x[: self.plant.dof])
        ee_pos.insert(0, self.plant.base)

        self.ee_poses.append(ee_pos)
        if len(self.ee_poses) > trail_len:
            self.ee_poses = np.delete(self.ee_poses, 0, 0).tolist()

        ani_plot_counter = 0
        # plot horizontal line
        if self.plot_horizontal_line:
            ll = 0.5 * (self.plant.l[0] + self.plant.l[1])
            self.animation_plots[ani_plot_counter].set_data(
                np.linspace(-ll, ll, 2),
                [self.horizontal_line_height, self.horizontal_line_height],
            )
            ani_plot_counter += 1

        # plot links
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(
                [ee_pos[link][0], ee_pos[link + 1][0]],
                [ee_pos[link][1], ee_pos[link + 1][1]],
            )
            ani_plot_counter += 1

        # plot base
        self.animation_plots[ani_plot_counter].set_data([ee_pos[0][0]], [ee_pos[0][1]])
        ani_plot_counter += 1

        # plot bodies
        for link in range(self.plant.n_links):
            self.animation_plots[ani_plot_counter].set_data(
                [ee_pos[link + 1][0]], [ee_pos[link + 1][1]]
            )
            ani_plot_counter += 1

            if self.plot_trail:
                self.animation_plots[ani_plot_counter].set_data(
                    [np.asarray(self.ee_poses)[:, link + 1, 0]],
                    [np.asarray(self.ee_poses)[:, link + 1, 1]],
                )
                ani_plot_counter += 1

            set_arrow_properties(
                self.tau_arrowarcs[link],
                self.tau_arrowheads[link],
                tau[link] / 5.0,
                ee_pos[link][0],
                ee_pos[link][1],
            )

        if self.plot_perturbations:
            for link in range(self.plant.n_links):
                set_arrow_properties(
                    self.tau_arrowarcs[self.plant.dof + link],
                    self.tau_arrowheads[self.plant.dof + link],
                    self.perturbation_array[link][dt_index] / 5.0,
                    ee_pos[link][0],
                    ee_pos[link][1],
                )

        if self.plot_inittraj:
            T, X, U = controller.get_init_trajectory()
            coords = []
            for x in X:
                coords.append(self.plant.forward_kinematics(x[: self.plant.dof])[-1])

            coords = np.asarray(coords)
            if len(coords) > 1:
                self.animation_plots[ani_plot_counter].set_data(
                    [coords.T[0]], [coords.T[1]]
                )
            ani_plot_counter += 1

        if self.plot_forecast:
            T, X, U = controller.get_forecast()
            coords = []
            for x in X:
                coords.append(self.plant.forward_kinematics(x[: self.plant.dof])[-1])

            coords = np.asarray(coords)
            if len(coords) > 1:
                self.animation_plots[ani_plot_counter].set_data(
                    [coords.T[0]], [coords.T[1]]
                )
            ani_plot_counter += 1

        t = float(self.animation_plots[ani_plot_counter].get_text()[4:])
        t = round(t + dt * sim_steps, 3)
        self.animation_plots[ani_plot_counter].set_text(f"t = {t}")

        # if the animation runs slower than real time
        # the time display will be red
        if not realtime:
            self.animation_plots[ani_plot_counter].set_color("red")
        else:
            self.animation_plots[ani_plot_counter].set_color("black")

        return self.animation_plots + self.tau_arrowarcs + self.tau_arrowheads

    def simulate_and_animate(
        self,
        t0,
        x0,
        tf,
        dt,
        controller=None,
        integrator="runge_kutta",
        plot_inittraj=False,
        plot_forecast=False,
        plot_trail=True,
        phase_plot=False,
        plot_perturbations=False,
        save_video=False,
        video_name="pendulum_swingup.mp4",
        anim_dt=0.02,
        plot_horizontal_line=False,
        horizontal_line_height=0.0,
        scale=1.0,
    ):
        """
        Simulate and animate the double pendulum for a time period under the
        control of a controller.
        The animation is only implemented for 2d serial chains.

        Parameters
        ----------
        t0 : float,
            start time, units=[s]
        x0 : array_like, shape=(4,), dtype=float,
            initial state of the double pendulum,
            order=[angle1, angle2, velocity1, velocity2],
            units=[rad, rad, rad/s, rad/s]
        tf : float
            final time, units=[s]
        dt : float
            timestep, unit=[s]
        controller : Controller object
            Controller whose control signal is used
            If None, motir torques are set to 0.
            (Default value = None)
        integrator : string
            string determining the integration method
            "euler" : Euler integrator
            "runge_kutta" : Runge Kutta integrator
             (Default value = "runge_kutta")
        plot_inittraj : bool
            Whether to plot an initial (reference) trajectory
            (Default value = False)
        plot_forecast : bool
            Whether to plot a forcasted trajectory
            (Default value = False)
        plot_trail : bool
            Whether to plot a trail for the masses
            (Default value = True)
        phase_plot : bool
            not used
            (Default value = False)
        save_video : bool
            Whether to render and save a video of the animation.
            Will be saved to video_name
            (Default value = False)
        video_name : string
            filepath where a video of the animation is stored if
            save_video==True
            (Default value = "pendulum_swingup.mp4")
        anim_dt : float
            timestep used for the animation, unit=[s]
            (Default value = 0.02)

        Returns
        -------
        list
            time points, unit=[s]
            shape=(N,)
        list
            shape=(N, 4)
            states, units=[rad, rad, rad/s, rad/s]
            order=[angle1, angle2, velocity1, velocity2]
        list
            shape=(N, 2)
            actuations/motor torques
            order=[u1, u2],
            units=[Nm]
        """

        self.plot_inittraj = plot_inittraj
        self.plot_forecast = plot_forecast
        self.plot_trail = plot_trail
        self.plot_horizontal_line = plot_horizontal_line
        self.horizontal_line_height = horizontal_line_height
        self.plot_perturbations = plot_perturbations
        # self.set_state(t0, x0)
        # self.reset_data_recorder()
        # self.record_data(t0, np.copy(x0), None)
        # self.meas_x_values.append(np.copy(x0))

        fig = plt.figure(figsize=(20 * scale, 20 * scale))
        self.animation_ax = plt.axes()
        self.animation_plots = []

        colors = ["#0077BE", "#f66338"]
        colors_trails = ["#d2eeff", "#ffebd8"]

        if self.plot_horizontal_line:
            (vl_plot,) = self.animation_ax.plot(
                [], [], "--", lw=2 * scale, color="black"
            )
            self.animation_plots.append(vl_plot)
        for link in range(self.plant.n_links):
            (bar_plot,) = self.animation_ax.plot([], [], "-", lw=10 * scale, color="k")
            self.animation_plots.append(bar_plot)

        (base_plot,) = self.animation_ax.plot(
            [], [], "s", markersize=25.0 * scale, color="black"
        )
        self.animation_plots.append(base_plot)

        for link in range(self.plant.n_links):
            (ee_plot,) = self.animation_ax.plot(
                [],
                [],
                "o",
                markersize=50.0 * scale,
                color="k",
                markerfacecolor=colors[link % len(colors)],
            )
            self.animation_plots.append(ee_plot)
            if self.plot_trail:
                (trail_plot,) = self.animation_ax.plot(
                    [],
                    [],
                    "-",
                    color=colors[link],
                    markersize=24 * scale,
                    markerfacecolor=colors_trails[link % len(colors_trails)],
                    lw=2 * scale,
                    markevery=10000,
                    markeredgecolor="None",
                )
                self.animation_plots.append(trail_plot)

        if self.plot_inittraj:
            (it_plot,) = self.animation_ax.plot(
                [], [], "--", lw=1 * scale, color="gray"
            )
            self.animation_plots.append(it_plot)
        if self.plot_forecast:
            (fc_plot,) = self.animation_ax.plot(
                [], [], "-", lw=1 * scale, color="green"
            )
            self.animation_plots.append(fc_plot)

        text_plot = self.animation_ax.text(
            0.1, 0.9, [], fontsize=60 * scale, transform=fig.transFigure
        )

        self.animation_plots.append(text_plot)

        num_steps = int((tf - t0) / anim_dt)
        self.par_dict = {}
        self.par_dict["dt"] = dt
        self.par_dict["x0"] = x0
        self.par_dict["t0"] = t0
        self.par_dict["anim_dt"] = anim_dt
        self.par_dict["controller"] = controller
        self.par_dict["integrator"] = integrator
        frames = num_steps * [self.par_dict]

        animation = FuncAnimation(
            fig,
            self._animation_step,
            frames=frames,
            init_func=self._animation_init,
            blit=True,
            repeat=False,
            interval=dt * 1000,
        )

        self.set_state(t0, x0)
        self.reset_data_recorder()
        self.record_data(t0, np.copy(x0), None)
        self.meas_x_values.append(x0)
        if save_video:
            print(f"Saving video to {video_name}")
            Writer = mplanimation.writers["ffmpeg"]
            writer = Writer(fps=60, bitrate=18000)
            animation.save(video_name, writer=writer)
            print("Saving video done.")
        else:
            plt.show()
        plt.close()

        return self.t_values, self.x_values, self.tau_values
