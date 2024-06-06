import os
import pickle
import yaml
import numpy as np

# from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.utils.csv_trajectory import load_trajectory
from double_pendulum.utils.lists import obj_to_list


class benchmarker:
    def __init__(
        self,
        controller,
        x0,
        dt,
        t_final,
        goal,
        epsilon=[0.1, 0.1, 1.0, 1.0],
        check_only_final_state=False,
        goal_check_method="epsilon",
        goal_check_method_height=0.9,
        integrator="runge_kutta",
    ):
        self.controller = controller
        self.x0 = np.asarray(x0)
        self.dt = dt
        self.t_final = t_final
        self.goal = np.asarray(goal)
        self.epsilon = epsilon
        self.check_only_final_state = check_only_final_state
        self.goal_check_method = goal_check_method
        self.goal_check_method_height = goal_check_method_height
        self.integrator = integrator

        self.mass = None
        self.length = None
        self.com = None
        self.damping = None
        self.gravity = None
        self.cfric = None
        self.inertia = None
        self.motor_inertia = None
        self.torque_limit = None

        self.plant = None
        self.simulator = None
        self.ref_trajectory = None

        self.Q = np.zeros((4, 4))
        self.R = np.zeros((2, 2))
        self.Qf = np.zeros((4, 4))

        self.traj_following = False
        self.t_traj = None
        self.x_traj = None
        self.u_traj = None

        self.ref_cost_free = None
        self.ref_cost_tf = None

    def set_model_parameter(
        self,
        mass=[0.608, 0.630],
        length=[0.3, 0.2],
        com=[0.275, 0.166],
        damping=[0.081, 0.0],
        cfric=[0.093, 0.186],
        gravity=9.81,
        inertia=[0.05472, 0.02522],
        motor_inertia=0.0,
        torque_limit=[0.0, 6.0],
        model_pars=None,
    ):
        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.gravity = gravity
        self.cfric = cfric
        self.inertia = inertia
        self.motor_inertia = motor_inertia
        self.torque_limit = torque_limit

        if model_pars is not None:
            self.mass = model_pars.m
            self.length = model_pars.l
            self.com = model_pars.r
            self.damping = model_pars.b
            self.cfric = model_pars.cf
            self.gravity = model_pars.g
            self.inertia = model_pars.I
            self.motor_inertia = model_pars.Ir
            # self.gr = model_pars.gr
            self.torque_limit = model_pars.tl

        # self.plant = SymbolicDoublePendulum(
        self.plant = DoublePendulumPlant(
            mass=self.mass,
            length=self.length,
            com=self.com,
            damping=self.damping,
            gravity=self.gravity,
            coulomb_fric=self.cfric,
            inertia=self.inertia,
            motor_inertia=self.motor_inertia,
            torque_limit=self.torque_limit,
        )

        self.simulator = Simulator(plant=self.plant)

    def set_init_traj(self, trajectory_csv):
        self.t_traj, self.x_traj, self.u_traj = load_trajectory(trajectory_csv)
        if len(self.u_traj) == len(self.x_traj):
            self.u_traj = self.u_traj[:-1]
        self.traj_following = True

    def set_cost_par(self, Q, R, Qf):
        self.Q = Q
        self.R = R
        self.Qf = Qf

    def compute_cost(self, x_traj, u_traj, mode="free"):
        len_traj = len(x_traj)

        if mode == "free":
            X = x_traj[:-1] - self.goal
            U = u_traj
            xf = x_traj[-1] - self.goal
        elif mode == "trajectory_following":
            n = min([len(x_traj), len(self.x_traj)])
            X = x_traj[:n] - self.x_traj[:n]
            nu = min([len(u_traj), len(self.u_traj)])
            U = u_traj[:nu] - self.u_traj[:nu]
            xf = x_traj[-1] - self.x_traj[-1]

        X_cost = np.einsum("jl, jk, lk", X.T, self.Q, X) / (len_traj - 1)
        U_cost = np.einsum("jl, jk, lk", U.T, self.R, U) / (len_traj - 1)
        Xf_cost = np.einsum("i, ij, j", xf, self.Qf, xf)

        cost = X_cost + U_cost + Xf_cost
        return cost

    def compute_ref_cost(self):
        if self.traj_following:
            self.ref_cost_free = self.compute_cost(
                self.x_traj, self.u_traj, mode="free"
            )
            self.ref_cost_tf = self.compute_cost(
                self.x_traj, self.u_traj, mode="trajectory_following"
            )

    def check_goal_success(self, x_traj):
        if self.goal_check_method == "epsilon":
            if self.check_only_final_state:
                lp = wrap_angles_top(x_traj[-1])
                pos1_succ = np.abs(lp[0] - self.goal[0]) < self.epsilon[0]
                pos2_succ = np.abs(lp[1] - self.goal[1]) < self.epsilon[1]
                vel1_succ = np.abs(lp[2] - self.goal[2]) < self.epsilon[2]
                vel2_succ = np.abs(lp[3] - self.goal[3]) < self.epsilon[3]
                succ = pos1_succ and pos2_succ and vel1_succ and vel2_succ
            else:
                succ = False
                for x in x_traj:
                    lp = wrap_angles_top(x)
                    pos1_succ = np.abs(lp[0] - self.goal[0]) < self.epsilon[0]
                    pos2_succ = np.abs(lp[1] - self.goal[1]) < self.epsilon[1]
                    vel1_succ = np.abs(lp[2] - self.goal[2]) < self.epsilon[2]
                    vel2_succ = np.abs(lp[3] - self.goal[3]) < self.epsilon[3]
                    succ = pos1_succ and pos2_succ and vel1_succ and vel2_succ
                    if succ:
                        break
        elif self.goal_check_method == "height":
            fk = self.plant.forward_kinematics(x_traj.T[:2])
            ee_pos_y = fk[1][1]

            goal_height = self.goal_check_method_height * (
                self.length[0] + self.length[1]
            )

            up = np.where(ee_pos_y > goal_height, True, False)

            if self.check_only_final_state:
                succ = True in up
            else:
                succ = up[-1]

        return succ

    def compute_success_measure(self, x_traj, u_traj):
        X = np.asarray(x_traj)
        U = np.asarray(u_traj)
        cost_free = self.compute_cost(X, U, mode="free")
        if self.traj_following:
            cost_tf = self.compute_cost(X, U, mode="trajectory_following")
        else:
            cost_tf = 0.0
        succ = self.check_goal_success(X)
        return cost_free, cost_tf, succ

    def simulate_and_get_cost(
        self,
        mass,
        length,
        com,
        damping,
        gravity,
        cfric,
        inertia,
        motor_inertia,
        torque_limit,
    ):
        # plant = SymbolicDoublePendulum(
        plant = DoublePendulumPlant(
            mass=mass,
            length=length,
            com=com,
            damping=damping,
            gravity=gravity,
            coulomb_fric=cfric,
            inertia=inertia,
            motor_inertia=motor_inertia,
            torque_limit=torque_limit,
        )

        simulator = Simulator(plant=plant)
        self.controller.reset()
        self.controller.init()

        T, X, U = simulator.simulate(
            t0=0.0,
            x0=self.x0,
            tf=self.t_final,
            dt=self.dt,
            controller=self.controller,
            integrator=self.integrator,
        )

        cost_free, cost_tf, succ = self.compute_success_measure(X, U)
        return cost_free, cost_tf, succ

    def check_modelpar_robustness(
        self,
        mpar_vars=["Ir", "m1r1", "I1", "b1", "cf1", "m2r2", "m2", "I2", "b2", "cf2"],
        var_lists={
            "Ir": [],
            "m1r1": [],
            "I1": [],
            "b1": [],
            "cf1": [],
            "m2r2": [],
            "m2": [],
            "I2": [],
            "b2": [],
            "cf2": [],
        },
    ):
        n_sims = 0
        for k in var_lists.keys():
            n_sims += len(var_lists[k])
        print(f"Computing model parameter robustness ({n_sims} simulations)")

        res_dict = {}
        for mp in mpar_vars:
            counter = 0
            nn_sims = len(var_lists[mp])
            print(
                "  Computing robustness to model parameter",
                mp,
                f" ({nn_sims} simulations)",
            )
            print(f"  {counter}/{nn_sims}", end="")
            C_free = []
            C_tf = []
            SUCC = []
            for var in var_lists[mp]:
                if mp == "Ir":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=self.inertia,
                        motor_inertia=var,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m1r1":
                    m1 = self.mass[0]
                    r1 = var / m1
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=[m1, self.mass[1]],
                        length=self.length,
                        com=[r1, self.com[1]],
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "I1":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=[var, self.inertia[1]],
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "b1":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=[var, self.damping[1]],
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "cf1":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=[var, self.cfric[1]],
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m2r2":
                    m2 = self.mass[1]
                    r2 = var / m2
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=[self.mass[0], m2],
                        length=self.length,
                        com=[self.com[0], r2],
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=[self.mass[0], var],
                        length=self.length,
                        com=self.com,
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "I2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=[self.inertia[0], var],
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "b2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=[self.damping[0], var],
                        gravity=self.gravity,
                        cfric=self.cfric,
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "cf2":
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                        mass=self.mass,
                        length=self.length,
                        com=self.com,
                        damping=self.damping,
                        gravity=self.gravity,
                        cfric=[self.cfric[0], var],
                        inertia=self.inertia,
                        motor_inertia=self.motor_inertia,
                        torque_limit=self.torque_limit,
                    )
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                counter += 1
                print("\r", end="")
                print(f"  {counter}/{nn_sims}", end="")
            res_dict[mp] = {}
            res_dict[mp]["values"] = var_lists[mp]
            res_dict[mp]["free_costs"] = C_free
            if self.traj_following:
                res_dict[mp]["following_costs"] = C_tf
            res_dict[mp]["successes"] = SUCC
            print("")
        return res_dict

    def check_perturbation_robustness(
        self,
        repetitions=20,
        n_pert_per_joint=3,
        min_t_dist=1.0,
        sigma_minmax=[0.01, 0.05],
        amplitude_min_max=[0.1, 1.0],
    ):
        n_sims = repetitions
        print(f"Computing pertubation robustness ({n_sims} simulations)")

        counter = 0
        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        mus = []
        sigmas = []
        amplitudes = []
        print(f"{counter}/{n_sims}", end="")
        for _ in range(repetitions):
            self.controller.reset()
            self.controller.init()
            (
                perturbation_array,
                mu,
                sigma,
                amplitude,
            ) = get_random_gauss_perturbation_array(
                self.t_final,
                self.dt,
                n_pert_per_joint,
                min_t_dist,
                sigma_minmax,
                amplitude_min_max,
            )
            self.simulator.set_disturbances(perturbation_array)
            T, X, U = self.simulator.simulate(
                t0=0.0,
                tf=self.t_final,
                dt=self.dt,
                x0=self.x0,
                controller=self.controller,
                integrator=self.integrator,
            )
            self.simulator.reset()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            mus.append(list(mu))
            sigmas.append(list(sigma))
            amplitudes.append(list(amplitude))
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
            counter += 1
            print("\r", end="")
            print(f"{counter}/{n_sims}", end="")
        res_dict["mu_list"] = mus
        res_dict["sigma_list"] = sigmas
        res_dict["amplitudes_list"] = amplitudes
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        print("")
        return res_dict

    def check_meas_noise_robustness(
        self,
        repetitions=10,
        meas_noise_mode="vel",
        meas_noise_sigma_list=[],
    ):
        # maybe add noise frequency
        # (on the real system noise frequency seems so be higher than
        # control frequency -> no frequency neccessary here)

        # leave for now for consitency with older data
        meas_noise_vfilters = ["None"]

        n_sims = repetitions * len(meas_noise_sigma_list) * len(meas_noise_vfilters)
        print(f"Computing noise robustness ({n_sims} simulations)")

        res_dict = {}
        for nf in meas_noise_vfilters:
            counter = 0
            nn_sims = repetitions * len(meas_noise_sigma_list)
            # print("  Using Noise filter: ", nf, f"({nn_sims} simulations)")
            print(f"  {counter}/{n_sims}", end="")
            C_free = []
            C_tf = []
            SUCC = []
            for na in meas_noise_sigma_list:
                rep_C_free = []
                rep_C_tf = []
                rep_SUCC = []
                for _ in range(repetitions):
                    self.controller.reset()
                    self.controller.init()
                    if meas_noise_mode == "posvel":
                        meas_noise_sigmas = [na, na, na, na]
                    elif meas_noise_mode == "vel":
                        meas_noise_sigmas = [0.0, 0.0, na, na]
                    self.simulator.set_measurement_parameters(
                        meas_noise_sigmas=meas_noise_sigmas
                    )
                    # self.simulator.set_filter_parameters(
                    #         meas_noise_cut=meas_noise_cut,
                    #         meas_noise_vfilter=nf,
                    #         meas_noise_vfilter_args=meas_noise_vfilter_args)
                    T, X, U = self.simulator.simulate(
                        t0=0.0,
                        tf=self.t_final,
                        dt=self.dt,
                        x0=self.x0,
                        controller=self.controller,
                        integrator=self.integrator,
                    )
                    self.simulator.reset()

                    cost_free, cost_tf, succ = self.compute_success_measure(X, U)
                    rep_C_free.append(cost_free)
                    rep_C_tf.append(cost_tf)
                    rep_SUCC.append(succ)
                    counter += 1
                    print("\r", end="")
                    print(f"  {counter}/{n_sims}", end="")
                C_free.append(rep_C_free)
                C_tf.append(rep_C_tf)
                SUCC.append(rep_SUCC)
            res_dict[nf] = {}
            res_dict[nf]["noise_sigma_list"] = meas_noise_sigma_list
            res_dict[nf]["free_costs"] = C_free
            if self.traj_following:
                res_dict[nf]["following_costs"] = C_tf
            res_dict[nf]["successes"] = SUCC
            res_dict[nf]["noise_mode"] = meas_noise_mode
            # res_dict[nf]["noise_cut"] = meas_noise_cut
            # res_dict[nf]["noise_vfilter"] = nf
            # res_dict[nf]["noise_vfilter_args"] = meas_noise_vfilter_args
            print("")
        return res_dict

    def check_unoise_robustness(self, repetitions=10, u_noise_sigma_list=[]):
        # maybe add noise frequency
        n_sims = repetitions * len(u_noise_sigma_list)
        print(f"Computing torque noise robustness ({n_sims} simulations)")

        counter = 0
        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        print(f"{counter}/{n_sims}", end="")
        for uns in u_noise_sigma_list:
            rep_C_free = []
            rep_C_tf = []
            rep_SUCC = []
            for _ in range(repetitions):
                self.controller.reset()
                self.controller.init()
                u_noise_sigmas = np.zeros(len(self.torque_limit))
                for i in range(len(self.torque_limit)):
                    if self.torque_limit[i] != 0.0:
                        u_noise_sigmas[i] = uns

                self.simulator.set_motor_parameters(u_noise_sigmas=u_noise_sigmas)
                T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator,
                )
                self.simulator.reset()

                cost_free, cost_tf, succ = self.compute_success_measure(X, U)
                rep_C_free.append(cost_free)
                rep_C_tf.append(cost_tf)
                rep_SUCC.append(succ)
                counter += 1
                print("\r", end="")
                print(f"{counter}/{n_sims}", end="")
            C_free.append(rep_C_free)
            C_tf.append(rep_C_tf)
            SUCC.append(rep_SUCC)
        res_dict["u_noise_sigma_list"] = u_noise_sigma_list
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        print("")
        return res_dict

    def check_uresponsiveness_robustness(self, u_responses=[]):
        n_sims = len(u_responses)
        print(f"Computing torque responsiveness robustness ({n_sims} simulations)")

        counter = 0
        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        print(f"{counter}/{n_sims}", end="")
        for ur in u_responses:
            self.controller.reset()
            self.controller.init()
            self.simulator.set_motor_parameters(u_responsiveness=ur)
            T, X, U = self.simulator.simulate(
                t0=0.0,
                tf=self.t_final,
                dt=self.dt,
                x0=self.x0,
                controller=self.controller,
                integrator=self.integrator,
            )
            self.simulator.reset()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
            counter += 1
            print("\r", end="")
            print(f"{counter}/{n_sims}", end="")
        res_dict["u_responsivenesses"] = u_responses
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        print("")
        return res_dict

    def check_delay_robustness(self, delay_mode="posvel", delays=[]):
        n_sims = len(delays)
        print(f"Computing delay robustness ({n_sims} simulations)")

        counter = 0
        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        print(f"{counter}/{n_sims}", end="")
        for de in delays:
            self.controller.reset()
            self.controller.init()
            self.simulator.set_measurement_parameters(delay=de, delay_mode=delay_mode)
            T, X, U = self.simulator.simulate(
                t0=0.0,
                tf=self.t_final,
                dt=self.dt,
                x0=self.x0,
                controller=self.controller,
                integrator=self.integrator,
            )
            self.simulator.reset()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
            counter += 1
            print("\r", end="")
            print(f"{counter}/{n_sims}", end="")
        res_dict["delay_mode"] = delay_mode
        res_dict["measurement_delay"] = delays
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        print("")
        return res_dict

    def benchmark(
        self,
        compute_model_robustness=True,
        compute_noise_robustness=True,
        compute_unoise_robustness=True,
        compute_uresponsiveness_robustness=True,
        compute_delay_robustness=True,
        compute_perturbation_robustness=False,
        mpar_vars=["Ir", "m1r1", "I1", "b1", "cf1", "m2r2", "m2", "I2", "b2", "cf2"],
        modelpar_var_lists={
            "Ir": [],
            "m1r1": [],
            "I1": [],
            "b1": [],
            "cf1": [],
            "m2r2": [],
            "m2": [],
            "I2": [],
            "b2": [],
            "cf2": [],
        },
        repetitions=10,
        meas_noise_mode="vel",
        meas_noise_sigma_list=[0.1, 0.3, 0.5],
        u_noise_sigma_list=[0.1, 0.5, 1.0],
        u_responses=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        delay_mode="vel",
        delays=[0.01, 0.02, 0.05, 0.1],
        perturbation_repetitions=50,
        perturbations_per_joint=3,
        perturbation_min_t_dist=1.0,
        perturbation_sigma_minmax=[0.01, 0.05],
        perturbation_amp_minmax=[0.1, 1.0],
    ):
        self.compute_model_robustness = compute_model_robustness
        self.compute_noise_robustness = compute_noise_robustness
        self.compute_unoise_robustness = compute_unoise_robustness
        self.compute_uresponsiveness_robustness = compute_uresponsiveness_robustness
        self.compute_delay_robustness = compute_delay_robustness
        self.compute_perturbation_robustness = compute_perturbation_robustness
        self.mpar_vars = mpar_vars
        self.modelpar_var_lists = modelpar_var_lists
        self.repetitions = repetitions
        self.meas_noise_mode = meas_noise_mode
        self.meas_noise_sigma_list = meas_noise_sigma_list
        self.u_noise_sigma_list = u_noise_sigma_list
        self.u_responses = u_responses
        self.delay_mode = delay_mode
        self.delays = delays
        self.perturbation_repetitions = perturbation_repetitions
        self.perturbations_per_joint = perturbations_per_joint
        self.perturbation_min_t_dist = perturbation_min_t_dist
        self.perturbation_sigma_minmax = perturbation_sigma_minmax
        self.perturbation_amp_minmax = perturbation_amp_minmax

        n_sims = 0
        if compute_model_robustness:
            for k in modelpar_var_lists.keys():
                n_sims += len(modelpar_var_lists[k])
        if compute_noise_robustness:
            n_sims += repetitions * len(meas_noise_sigma_list)
        if compute_unoise_robustness:
            n_sims += repetitions * len(u_noise_sigma_list)
        if compute_uresponsiveness_robustness:
            n_sims += len(u_responses)
        if compute_delay_robustness:
            n_sims += len(delays)
        if compute_perturbation_robustness:
            n_sims += perturbation_repetitions
        print(
            f"\nWill in total compute {n_sims} simulations for testing the robustness of the controller\n"
        )

        self.res = {}
        if compute_model_robustness:
            res_model = self.check_modelpar_robustness(
                mpar_vars=mpar_vars, var_lists=modelpar_var_lists
            )
            self.res["model_robustness"] = res_model
        if compute_noise_robustness:
            res_noise = self.check_meas_noise_robustness(
                repetitions=repetitions,
                meas_noise_mode=meas_noise_mode,
                meas_noise_sigma_list=meas_noise_sigma_list,
                # meas_noise_cut=meas_noise_cut,
                # meas_noise_vfilters=meas_noise_vfilters,
                # meas_noise_vfilter_args=meas_noise_vfilter_args,
            )
            self.res["meas_noise_robustness"] = res_noise
        if compute_unoise_robustness:
            res_unoise = self.check_unoise_robustness(
                u_noise_sigma_list=u_noise_sigma_list
            )
            self.res["u_noise_robustness"] = res_unoise
        if compute_uresponsiveness_robustness:
            res_uresp = self.check_uresponsiveness_robustness(u_responses=u_responses)
            self.res["u_responsiveness_robustness"] = res_uresp
        if compute_delay_robustness:
            res_delay = self.check_delay_robustness(
                delay_mode=delay_mode, delays=delays
            )
            self.res["delay_robustness"] = res_delay
        if compute_perturbation_robustness:
            res_pertub = self.check_perturbation_robustness(
                repetitions=perturbation_repetitions,
                n_pert_per_joint=perturbations_per_joint,
                min_t_dist=perturbation_min_t_dist,
                sigma_minmax=perturbation_sigma_minmax,
                amplitude_min_max=perturbation_amp_minmax,
            )
            self.res["perturbation_robustness"] = res_pertub
        return self.res

    def save(self, save_dir="."):
        f = open(os.path.join(save_dir, "benchmark_results.pkl"), "wb")
        pickle.dump(self.res, f)
        f.close()

        modelpar_var_lists_save = self.modelpar_var_lists

        for key in modelpar_var_lists_save.keys():
            modelpar_var_lists_save[key] = obj_to_list(modelpar_var_lists_save[key])

        par_dict = {
            "x0": obj_to_list(self.x0),
            "dt": self.dt,
            "t_final": self.t_final,
            "goal": obj_to_list(self.goal),
            "epsilon": list(self.epsilon),
            "check_only_final_state": self.check_only_final_state,
            "goal_check_method": self.goal_check_method,
            "goal_check_method_height": self.goal_check_method_height,
            "integrator": self.integrator,
            "Q": obj_to_list(self.Q),
            "Qf": obj_to_list(self.Qf),
            "R": obj_to_list(self.R),
            "traj_following": self.traj_following,
            "compute_model_robustness": self.compute_model_robustness,
            "compute_noise_robustness": self.compute_noise_robustness,
            "compute_unoise_robustness": self.compute_unoise_robustness,
            "compute_uresponsiveness_robustness": self.compute_uresponsiveness_robustness,
            "compute_delay_robustness": self.compute_delay_robustness,
            "compute_perturbation_robustness": self.compute_perturbation_robustness,
            "mpar_vars": self.mpar_vars,
            "modelpar_var_lists": modelpar_var_lists_save,
            "repetitions": self.repetitions,
            "meas_noise_mode": self.meas_noise_mode,
            "meas_noise_sigma_list": obj_to_list(self.meas_noise_sigma_list),
            "u_noise_sigma_list": obj_to_list(self.u_noise_sigma_list),
            "u_responses": obj_to_list(self.u_responses),
            "delay_mode": self.delay_mode,
            "delays": obj_to_list(self.delays),
            "perturbation_repetitions": self.perturbation_repetitions,
            "perturbations_per_joint": self.perturbations_per_joint,
            "perturbation_min_t_dist": self.perturbation_min_t_dist,
            "perturbation_sigma_minmax": list(self.perturbation_sigma_minmax),
            "perturbation_amp_minmax": list(self.perturbation_amp_minmax),
        }

        with open(os.path.join(save_dir, "benchmark_parameters.yml"), "w") as f:
            yaml.dump(par_dict, f)
