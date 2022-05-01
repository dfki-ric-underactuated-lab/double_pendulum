import os
import numpy as np
import pandas as pd

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


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
        self.x0 = x0
        self.dt = dt
        self.t_final = t_final
        self.goal = goal
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

        self.t_traj = None
        self.x_traj = None
        self.u_traj = None

        self.ref_cost_free = None
        self.ref_cost_tf = None

    def set_model_parameter(self,
                            mass,
                            length,
                            com,
                            damping,
                            gravity,
                            cfric,
                            inertia,
                            motor_inertia,
                            torque_limit):

        self.mass = mass
        self.length = length
        self.com = com
        self.damping = damping
        self.gravity = gravity
        self.cfric = cfric
        self.inertia = inertia
        self.motor_inertia = motor_inertia
        self.torque_limit = torque_limit

        self.plant = SymbolicDoublePendulum(mass=mass,
                                            length=length,
                                            com=com,
                                            damping=damping,
                                            gravity=gravity,
                                            coulomb_fric=cfric,
                                            inertia=inertia,
                                            motor_inertia=motor_inertia,
                                            torque_limit=torque_limit)

        self.simulator = Simulator(plant=self.plant)

    def set_init_traj(self, trajectory_csv, read_with):
        if read_with == "pandas":
            self.ref_trajectory = pd.read_csv(trajectory_csv)

            time_traj = np.asarray(self.ref_trajectory["time"])
            pos1_traj = np.asarray(self.ref_trajectory["shoulder_pos"])
            pos2_traj = np.asarray(self.ref_trajectory["elbow_pos"])
            vel1_traj = np.asarray(self.ref_trajectory["shoulder_vel"])
            vel2_traj = np.asarray(self.ref_trajectory["elbow_vel"])
            tau1_traj = np.asarray(self.ref_trajectory["shoulder_torque"])
            tau2_traj = np.asarray(self.ref_trajectory["elbow_torque"])

        elif read_with == "numpy":
            self.ref_trajectory = np.loadtxt(trajectory_csv, skiprows=1, delimiter=",")

            time_traj = self.ref_trajectory[:, 0]
            pos1_traj = self.ref_trajectory[:, 1]
            pos2_traj = self.ref_trajectory[:, 2]
            vel1_traj = self.ref_trajectory[:, 3]
            vel2_traj = self.ref_trajectory[:, 4]
            tau1_traj = self.ref_trajectory[:, 5]
            tau2_traj = self.ref_trajectory[:, 6]

        self.t_traj = time_traj.T
        self.x_traj = np.asarray([pos1_traj, pos2_traj,
                                  vel1_traj, vel2_traj]).T
        self.u_traj = np.asarray([tau1_traj, tau2_traj]).T

    def set_cost_par(self, Q, R, Qf):

        self.Q = Q
        self.R = R
        self.Qf = Qf

    def compute_cost(self, x_traj, u_traj, mode="free"):

        if mode == "free":
            X = x_traj[:-1] - self.goal
            U = u_traj
            xf = x_traj[-1] - self.goal
        elif mode == "trajectory_following":
            X = x_traj[:-1] - self.x_traj[:-1]
            U = u_traj - self.u_traj
            xf = x_traj[-1] - self.x_traj[-1]

        X_cost = np.einsum('jl, jk, lk', X.T, self.Q, X)
        U_cost = np.einsum('jl, jk, lk', U.T, self.R, U)
        Xf_cost = np.einsum('i, ij, j', xf, self.Qf, xf)

        cost = X_cost + U_cost + Xf_cost
        return cost

    def compute_ref_cost(self):
        self.ref_cost_free = self.compute_cost(self.x_traj, self.u_traj, mode="free")
        self.ref_cost_tf = self.compute_cost(self.x_traj, self.u_traj, mode="trajectory_following")

    def check_goal_success(self, x_traj, eps=0.1):
        succ = np.max(np.diff(x_traj[-1] - self.goal)) < eps
        return succ

    def compute_success_measure(self, x_traj, u_traj):
        X = np.asarray(x_traj)
        U = np.asarray(u_traj)
        cost_free = self.compute_cost(X, U, mode="free")
        cost_tf = self.compute_cost(X, U, mode="trajectory_following")
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

    def check_modelpar_variation(self,
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
            res_dict[mp]["free_costs"] = C_free
            res_dict[mp]["following_costs"] = C_free
            res_dict[mp]["successes"] = SUCC
        return res_dict

    def benchmark(self,
                  compute_model_robustness=True,
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
                                      "cf2": []}):

        res = {}
        if compute_model_robustness:
            res_model = self.check_modelpar_variation(
                    mpar_vars=mpar_vars,
                    var_lists=modelpar_var_lists)
            res["model_robustness"] = res_model
        return res
