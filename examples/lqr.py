import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries


robot = "pendubot"

# tmotors
# mass = [0.608, 0.630]
# length = [0.3, 0.2]
# com = [0.275, 0.166]
# damping = [0.081, 0.0]
# # cfric = [0.093, 0.186]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.05472, 0.02522]
# torque_limit = [0.0, 4.0]

# mjbots
# mass = [0.608, 0.6268]
# length = [0.3, 0.2]
# com = [0.317763, 0.186981]
# damping = [7.480e-02, 1.602e-08]
# # cfric = [7.998e-02, 6.336e-02]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.0280354, 0.0183778]
# torque_limit = [0.0, 10.0]

mass = [0.608, 0.22]
length = [0.2, 0.4]
com = [length[0], length[1]]
damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2., mass[1]*length[1]**2.]
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

# simulation parameters
dt = 0.005
t_final = 5.0
if robot == "acrobot":
    x0 = [2.85, 0.7, 0.0, 0.0]
if robot == "pendubot":
    x0 = [2.9, 0.3, 0.0, 0.0]

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)


controller = LQRController(mass=mass,
                           length=length,
                           com=com,
                           damping=damping,
                           gravity=gravity,
                           coulomb_fric=cfric,
                           inertia=inertia,
                           torque_limit=torque_limit)

controller.set_goal([np.pi, 0., 0., 0.])
if robot == "acrobot":
    # 11.67044417  0.10076462  3.86730188  0.11016735  0.18307841
    # [6.60845661 0.0203263  1.59384209 0.11347305 0.08903846]
    # 1.649609700742603735e+01 9.094310297259731612e+01 7.128663050519863653e-02 1.116726623434960083e-02 3.647472178659907360e+00
    #c[7.568227051118126880e+00, 1.851805841833500610e+00, 9.989157089721836247e-01, 9.994476149737525628e-01, 7.567700329462909714e-01]
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
elif robot == "pendubot":
    # [0.01251931 6.51187283 6.87772744 9.35785251 0.02354949]
    # [8.74006242e+01 1.12451099e-02 9.59966065e+01 8.99725246e-01 2.37517689e-01]
    # [1.16402700e+01 7.95782007e+01 7.29021272e-02 3.02202319e-04 1.29619149e-01]
    Q = np.diag((11.64, 79.58, 0.073, 0.0003))
    R = np.diag((0.13, 0.13))

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
                          cost_to_go_cut=15.0)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   save_video=False,
                                   video_name="data/"+robot+"/lqr/acrobot_lqr")

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]])
