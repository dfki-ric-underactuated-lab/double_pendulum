import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.simulation.simulation import Simulator


# TODO: interface for combined optimization is defined 2 times
class lqr_check_ctg:
    def __init__(self, p, c, tf=5, dt=0.01):
        self.tf = tf
        self.dt = dt
        self.sim = Simulator(plant=p)
        self.controller = c

    def sim_callback(self, x0):
        """
        Callback that is passed to the probabilitic RoA estimation class
        """
        t, x, tau = self.sim.simulate(
                t0=0.0,
                x0=x0,
                tf=self.tf,
                dt=self.dt,
                controller=self.controller,
                integrator="runge_kutta")

        if np.isnan(x[-1]).all():
            return False
        else:
            return True


class lqr_check_epsilon:
    def __init__(self, p, c, tf=5, dt=0.01,
                 goal=np.array([np.pi, 0., 0., 0.]), eps_p=0.01, eps_v=0.05):
        self.tf = tf
        self.dt = dt
        self.sim = Simulator(plant=p)
        self.controller = c
        self.goal = goal
        self.eps_p = eps_p
        self.eps_v = eps_v

    def sim_callback(self, x0):
        T, X, U = self.sim.simulate(
                t0=0.0,
                x0=x0,
                tf=self.tf,
                dt=self.dt,
                controller=self.controller,
                integrator="runge_kutta")

        pos_check = np.max(np.abs(np.asarray(X)[-1][:2] - self.goal[:2])) <= self.eps_p
        vel_check = np.max(np.abs(np.asarray(X)[-1][2:] - self.goal[2:])) <= self.eps_v

        ret = pos_check and vel_check
        return ret


def lqr_check_ctg_verification(dpar, cpar, x0, grid, idx1, idx2):
    mass = [0.608, dpar[0]]
    length = [dpar[1], dpar[2]]
    com = [length[0], length[1]]
    damping = [0.0, 0.0]
    cfric = [0., 0.]
    gravity = 9.81
    inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
    torque_limit = [0.0, 5.0]

    goal = np.array([np.pi, 0., 0., 0.])

    plant = SymbolicDoublePendulum(mass=mass,
                                   length=length,
                                   com=com,
                                   damping=damping,
                                   gravity=gravity,
                                   coulomb_fric=cfric,
                                   inertia=inertia,
                                   torque_limit=torque_limit)
    dt = 0.01
    t_final = 5.0

    controller = LQRController(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

    controller.set_goal(goal)
    Q = np.diag((cpar[0], cpar[1], cpar[2], cpar[3]))
    R = np.diag((cpar[4], cpar[4]))
    controller.set_cost_parameters(p1p1_cost=Q[0, 0],
                                   p2p2_cost=Q[1, 1],
                                   v1v1_cost=Q[2, 2],
                                   v2v2_cost=Q[3, 3],
                                   p1v1_cost=0.,
                                   p1v2_cost=0.,
                                   p2v1_cost=0.,
                                   p2v2_cost=0.,
                                   u1u1_cost=R[0, 0],
                                   u2u2_cost=R[1, 1],
                                   u1u2_cost=0.)
    # controller.set_parameters(failure_value=0.0,
    #                           cost_to_go_cut=np.inf)
    controller.init()

    lcc = lqr_check_ctg(plant, controller, t_final, dt)
    ret = lcc.sim_callback(x0)
    grid[str(idx1).zfill(2)+"_"+str(idx2).zfill(2)] = ret


def lqr_check_epsilon_verification(dpar, cpar, x0, grid, idx1,
                                   idx2, eps_p=0.01, eps_v=0.05):

    mass = [0.608, dpar[0]]
    length = [dpar[1], dpar[2]]
    com = [length[0], length[1]]
    damping = [0.0, 0.0]
    cfric = [0., 0.]
    gravity = 9.81
    inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
    torque_limit = [0.0, 5.0]

    goal = np.array([np.pi, 0., 0., 0.])

    plant = SymbolicDoublePendulum(mass=mass,
                                   length=length,
                                   com=com,
                                   damping=damping,
                                   gravity=gravity,
                                   coulomb_fric=cfric,
                                   inertia=inertia,
                                   torque_limit=torque_limit)

    sim = Simulator(plant=plant)
    dt = 0.01
    t_final = 5.0

    controller = LQRController(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

    controller.set_goal(goal)
    Q = np.diag((cpar[0], cpar[1], cpar[2], cpar[3]))
    R = np.diag((cpar[4], cpar[4]))
    controller.set_cost_parameters(p1p1_cost=Q[0, 0],
                                   p2p2_cost=Q[1, 1],
                                   v1v1_cost=Q[2, 2],
                                   v2v2_cost=Q[3, 3],
                                   p1v1_cost=0.,
                                   p1v2_cost=0.,
                                   p2v1_cost=0.,
                                   p2v2_cost=0.,
                                   u1u1_cost=R[0, 0],
                                   u2u2_cost=R[1, 1],
                                   u1u2_cost=0.)
    controller.set_parameters(failure_value=0.0,
                              cost_to_go_cut=np.inf)
    controller.init()
    T, X, U = sim.simulate(t0=0.0, x0=x0, tf=t_final, dt=dt,
                           controller=controller, integrator="runge_kutta")

    pos_check = np.max(np.abs(np.asarray(X)[-1][:2] - goal[:2])) <= eps_p
    vel_check = np.max(np.abs(np.asarray(X)[-1][2:] - goal[2:])) <= eps_v

    ret = pos_check and vel_check
    grid[str(idx1).zfill(2)+"_"+str(idx2).zfill(2)] = ret
