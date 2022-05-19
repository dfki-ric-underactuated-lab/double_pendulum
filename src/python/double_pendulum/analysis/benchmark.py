import numpy as np

from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import load_trajectory


class benchmarker():
    def __init__(self,
                 controller,
                 x0,
                 dt,
                 t_final,
                 goal,
                 integrator="runge_kutta",
                 save_dir="benchmark"):
        self.controller = controller
        self.x0 = np.asarray(x0)
        self.dt = dt
        self.t_final = t_final
        self.goal = np.asarray(goal)
        self.integrator = integrator
        self.save_dir = save_dir

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

        self.Q = None
        self.R = None
        self.Qf = None

        self.traj_following = False
        self.t_traj = None
        self.x_traj = None
        self.u_traj = None

        self.ref_cost_free = None
        self.ref_cost_tf = None

    def set_model_parameter(self,
                            mass=[0.608, 0.630],
                            length=[0.3, 0.2],
                            com=[0.275, 0.166],
                            damping=[0.081, 0.0],
                            cfric=[0.093, 0.186],
                            gravity=9.81,
                            inertia=[0.05472, 0.02522],
                            motor_inertia=0.,
                            torque_limit=[0.0, 6.0],
                            model_pars=None):

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

        self.plant = SymbolicDoublePendulum(mass=self.mass,
                                            length=self.length,
                                            com=self.com,
                                            damping=self.damping,
                                            gravity=self.gravity,
                                            coulomb_fric=self.cfric,
                                            inertia=self.inertia,
                                            motor_inertia=self.motor_inertia,
                                            torque_limit=self.torque_limit)

        self.simulator = Simulator(plant=self.plant)

    def set_init_traj(self, trajectory_csv, read_with):
        self.t_traj, self.x_traj, self.u_traj = load_trajectory(trajectory_csv, read_with)
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
            X = x_traj[:-1] - self.x_traj[:-1]
            U = u_traj - self.u_traj
            xf = x_traj[-1] - self.x_traj[-1]

        X_cost = np.einsum('jl, jk, lk', X.T, self.Q, X) / (len_traj-1)
        U_cost = np.einsum('jl, jk, lk', U.T, self.R, U) / (len_traj-1)
        Xf_cost = np.einsum('i, ij, j', xf, self.Qf, xf)

        cost = X_cost + U_cost + Xf_cost
        return cost

    def compute_ref_cost(self):
        if self.traj_following:
            self.ref_cost_free = self.compute_cost(self.x_traj, self.u_traj, mode="free")
            self.ref_cost_tf = self.compute_cost(self.x_traj, self.u_traj, mode="trajectory_following")

    def check_goal_success(self, x_traj, pos_eps=0.1, vel_eps=0.5):
        lp = wrap_angles_top(x_traj[-1])
        pos_succ = np.max(lp[:2] - self.goal[:2]) < pos_eps
        vel_succ = np.max(lp[2:] - self.goal[2:]) < vel_eps
        return (pos_succ and vel_succ)

    def compute_success_measure(self, x_traj, u_traj):
        X = np.asarray(x_traj)
        U = np.asarray(u_traj)
        cost_free = self.compute_cost(X, U, mode="free")
        if self.traj_following:
            cost_tf = self.compute_cost(X, U, mode="trajectory_following")
        else:
            cost_tf = 0.
        succ = self.check_goal_success(X)
        return cost_free, cost_tf, succ

    def simulate_and_get_cost(self,
                              mass,
                              length,
                              com,
                              damping,
                              gravity,
                              cfric,
                              inertia,
                              motor_inertia,
                              torque_limit):

        plant = SymbolicDoublePendulum(mass=mass,
                                       length=length,
                                       com=com,
                                       damping=damping,
                                       gravity=gravity,
                                       coulomb_fric=cfric,
                                       inertia=inertia,
                                       motor_inertia=motor_inertia,
                                       torque_limit=torque_limit)

        simulator = Simulator(plant=plant)
        self.controller.init()

        T, X, U = simulator.simulate(t0=0., x0=self.x0, tf=self.t_final,
                                     dt=self.dt, controller=self.controller,
                                     integrator=self.integrator)

        cost_free, cost_tf, succ = self.compute_success_measure(X, U)
        return cost_free, cost_tf, succ

    def check_modelpar_robustness(
            self,
            mpar_vars=["Ir",
                       "m1r1", "I1", "b1", "cf1",
                       "m2r2", "m2", "I2", "b2", "cf2"],
            var_lists={"Ir": [],
                       "m1r1": [],
                       "I1": [],
                       "b1": [],
                       "cf1": [],
                       "m2r2": [],
                       "m2": [],
                       "I2": [],
                       "b2": [],
                       "cf2": []},
            ):

        print("computing model parameter robustness...")

        res_dict = {}
        for mp in mpar_vars:
            print("  ", mp)
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
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m1r1":
                    m1 = self.mass[0]
                    r1 = var/m1
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=[m1, self.mass[1]],
                            length=self.length,
                            com=[r1, self.com[1]],
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
                elif mp == "m2r2":
                    m2 = self.mass[1]
                    r2 = var/m2
                    cost_free, cost_tf, succ = self.simulate_and_get_cost(
                            mass=[self.mass[0], m2],
                            length=self.length,
                            com=[self.com[0], r2],
                            damping=self.damping,
                            gravity=self.gravity,
                            cfric=self.cfric,
                            inertia=self.inertia,
                            motor_inertia=self.motor_inertia,
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
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
                            torque_limit=self.torque_limit)
                    C_free.append(cost_free)
                    C_tf.append(cost_tf)
                    SUCC.append(succ)
            res_dict[mp] = {}
            res_dict[mp]["values"] = var_lists[mp]
            res_dict[mp]["free_costs"] = C_free
            if self.traj_following:
                res_dict[mp]["following_costs"] = C_tf
            res_dict[mp]["successes"] = SUCC
        return res_dict

    def check_perturbation_robustness(self,
                                      time_stamps=[],
                                      tau_perts=[]):
        pass

    def check_meas_noise_robustness(self,
                                    repetitions=10,
                                    meas_noise_mode="vel",
                                    meas_noise_sigma_list=[],
                                    meas_noise_cut=0.,
                                    meas_noise_vfilters=["None", "lowpass", "kalman"],
                                    meas_noise_vfilter_args={"alpha": 0.3}):
        # maybe add noise frequency
        # (on the real system noise frequency seems so be higher than
        # control frequency -> no frequency neccessary here)
        print("computing noise robustness...")

        res_dict = {}
        for nf in meas_noise_vfilters:
            print("  ", nf)
            C_free = []
            C_tf = []
            SUCC = []
            for na in meas_noise_sigma_list:
                rep_C_free = []
                rep_C_tf = []
                rep_SUCC = []
                for _ in range(repetitions):
                    self.controller.init()
                    if meas_noise_mode == "posvel":
                        meas_noise_sigmas = [na, na, na, na]
                    elif meas_noise_mode == "vel":
                        meas_noise_sigmas = [0., 0., na, na]
                    self.simulator.set_measurement_parameters(
                            meas_noise_sigmas=meas_noise_sigmas)
                    self.simulator.set_filter_parameters(
                            meas_noise_cut=meas_noise_cut,
                            meas_noise_vfilter=nf,
                            meas_noise_vfilter_args=meas_noise_vfilter_args)
                    T, X, U = self.simulator.simulate(
                            t0=0.0,
                            tf=self.t_final,
                            dt=self.dt,
                            x0=self.x0,
                            controller=self.controller,
                            integrator=self.integrator)
                    self.simulator.reset()

                    cost_free, cost_tf, succ = self.compute_success_measure(X, U)
                    rep_C_free.append(cost_free)
                    rep_C_tf.append(cost_tf)
                    rep_SUCC.append(succ)
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
            res_dict[nf]["noise_cut"] = meas_noise_cut
            res_dict[nf]["noise_vfilter"] = nf
            res_dict[nf]["noise_vfilter_args"] = meas_noise_vfilter_args
        return res_dict

    def check_unoise_robustness(self,
                                repetitions=10,
                                u_noise_sigma_list=[]):
        # maybe add noise frequency
        print("computing torque noise robustness...")

        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for uns in u_noise_sigma_list:
            rep_C_free = []
            rep_C_tf = []
            rep_SUCC = []
            for _ in range(repetitions):
                self.controller.init()
                u_noise_sigmas = np.zeros(len(self.torque_limit))
                for i in range(len(self.torque_limit)):
                    if self.torque_limit[i] != 0.:
                        u_noise_sigmas[i] = uns

                self.simulator.set_motor_parameters(u_noise_sigmas=u_noise_sigmas)
                T, X, U = self.simulator.simulate(
                        t0=0.0,
                        tf=self.t_final,
                        dt=self.dt,
                        x0=self.x0,
                        controller=self.controller,
                        integrator=self.integrator)
                self.simulator.reset()

                cost_free, cost_tf, succ = self.compute_success_measure(X, U)
                rep_C_free.append(cost_free)
                rep_C_tf.append(cost_tf)
                rep_SUCC.append(succ)
            C_free.append(rep_C_free)
            C_tf.append(rep_C_tf)
            SUCC.append(rep_SUCC)
        res_dict["u_noise_sigma_list"] = u_noise_sigma_list
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def check_uresponsiveness_robustness(self,
                                         u_responses=[]):
        print("computing torque responsiveness robustness...")

        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for ur in u_responses:
            self.controller.init()
            self.simulator.set_motor_parameters(u_responsiveness=ur)
            T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator)
            self.simulator.reset()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
        res_dict["u_responsivenesses"] = u_responses
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def check_delay_robustness(self,
                               delay_mode="posvel",
                               delays=[]):
        print("computing delay robustness...")
        res_dict = {}
        C_free = []
        C_tf = []
        SUCC = []
        for de in delays:
            self.controller.init()
            self.simulator.set_measurement_parameters(
                    delay=de,
                    delay_mode=delay_mode)
            T, X, U = self.simulator.simulate(
                    t0=0.0,
                    tf=self.t_final,
                    dt=self.dt,
                    x0=self.x0,
                    controller=self.controller,
                    integrator=self.integrator)
            self.simulator.reset()

            cost_free, cost_tf, succ = self.compute_success_measure(X, U)
            C_free.append(cost_free)
            C_tf.append(cost_tf)
            SUCC.append(succ)
        res_dict["delay_mode"] = delay_mode
        res_dict["measurement_delay"] = delays
        res_dict["free_costs"] = C_free
        if self.traj_following:
            res_dict["following_costs"] = C_tf
        res_dict["successes"] = SUCC
        return res_dict

    def benchmark(self,
                  compute_model_robustness=True,
                  compute_noise_robustness=True,
                  compute_unoise_robustness=True,
                  compute_uresponsiveness_robustness=True,
                  compute_delay_robustness=True,
                  mpar_vars=["Ir",
                             "m1r1", "I1", "b1", "cf1",
                             "m2r2", "m2", "I2", "b2", "cf2"],
                  modelpar_var_lists={"Ir": [],
                                      "m1r1": [],
                                      "I1": [],
                                      "b1": [],
                                      "cf1": [],
                                      "m2r2": [],
                                      "m2": [],
                                      "I2": [],
                                      "b2": [],
                                      "cf2": []},
                  repetitions=10,
                  meas_noise_mode="vel",
                  meas_noise_sigma_list=[0.1, 0.3, 0.5],
                  meas_noise_cut=0.5,
                  meas_noise_vfilters=["lowpass"],
                  meas_noise_vfilter_args={"alpha": 0.3},
                  u_noise_sigma_list=[0.1, 0.5, 1.0],
                  u_responses=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                  delay_mode="vel",
                  delays=[0.01, 0.02, 0.05, 0.1]):

        res = {}
        if compute_model_robustness:
            res_model = self.check_modelpar_robustness(
                    mpar_vars=mpar_vars,
                    var_lists=modelpar_var_lists)
            res["model_robustness"] = res_model
        if compute_noise_robustness:
            res_noise = self.check_meas_noise_robustness(
                    repetitions=repetitions,
                    meas_noise_mode=meas_noise_mode,
                    meas_noise_sigma_list=meas_noise_sigma_list,
                    meas_noise_cut=meas_noise_cut,
                    meas_noise_vfilters=meas_noise_vfilters,
                    meas_noise_vfilter_args=meas_noise_vfilter_args)
            res["meas_noise_robustness"] = res_noise
        if compute_unoise_robustness:
            res_unoise = self.check_unoise_robustness(
                    u_noise_sigma_list=u_noise_sigma_list)
            res["u_noise_robustness"] = res_unoise
        if compute_uresponsiveness_robustness:
            res_uresp = self.check_uresponsiveness_robustness(
                    u_responses=u_responses)
            res["u_responsiveness_robustness"] = res_uresp
        if compute_delay_robustness:
            res_delay = self.check_delay_robustness(
                    delay_mode=delay_mode,
                    delays=delays)
            res["delay_robustness"] = res_delay
        return res
