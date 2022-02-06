import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import (SymbolicPFLController,
                                                                                    SymbolicPFLAndLQRController)


robot = "pendubot"
pfl_method = "noncollocated"
with_cfric = False
with_lqr = True

x0 = [0.1, 0.0, 0.0, 0.0]
dt = 0.01
t_final = 30.0

if robot == "acrobot":
    if pfl_method == "collocated":
        if with_cfric:
            par = [9.94271982, 1.56306923, 3.27636175]  # ok
        else:
            par = [9.94246152, 9.84124115, 9.81120166]  # good
    elif pfl_method == "noncollocated":
        par = [9.19534629, 2.24529733, 5.90567362]  # good
    else:
        print(f"pfl_method {pfl_method} not found. Please set eigher" +
              "pfl_method='collocated' or pfl_method='noncollocated'")
        exit()
elif robot == "pendubot":
    if pfl_method == "collocated":
        par = [6.97474837, 9.84031538, 9.1297417]  # bad
    elif pfl_method == "noncollocated":
        # par = [6.97474837, 9.84031538, 9.1297417]  # bad
        # par = [4.91129641, 10., 1.64418209]
        par = [26.34039456, 99.99876263, 11.89097532]
        # par = [15.64747394, 19.5291726, 3.71447987]
    else:
        print(f"pfl_method {pfl_method} not found. Please set eigher" +
              "pfl_method='collocated' or pfl_method='noncollocated'")
        exit()
else:
    print(f"robot {robot} not found. Please set eigher" +
          "robot='acrobot' or robot='pendubot'")
    exit()

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
if with_cfric:
    cfric = [0.093, 0.186]
else:
    cfric = [0.0, 0.0]
gravity = 9.81
inertia = [0.05472, 0.2522]
if robot == "acrobot":
    torque_limit = [0.0, 3.0]
if robot == "pendubot":
    torque_limit = [3.0, 0.0]

double_pendulum = SymbolicDoublePendulum(mass=mass,
                                         length=length,
                                         com=com,
                                         damping=damping,
                                         gravity=gravity,
                                         coulomb_fric=cfric,
                                         inertia=inertia,
                                         torque_limit=torque_limit)

if with_lqr:
    controller = SymbolicPFLAndLQRController(mass,
                                             length,
                                             com,
                                             damping,
                                             gravity,
                                             cfric,
                                             inertia,
                                             torque_limit,
                                             robot,
                                             pfl_method)
    if robot == "acrobot":
        controller.lqr_controller.set_cost_parameters(p1p1_cost=11.67,
                                                      p2p2_cost=3.87,
                                                      v1v1_cost=0.10,
                                                      v2v2_cost=0.11,
                                                      p1v1_cost=0.,
                                                      p1v2_cost=0.,
                                                      p2v1_cost=0.,
                                                      p2v2_cost=0.,
                                                      u1u1_cost=0.18,
                                                      u2u2_cost=0.18,
                                                      u1u2_cost=0.)
    else:
        controller.lqr_controller.set_cost_parameters(p1p1_cost=0.01251931,
                                                      p2p2_cost=6.87772744,
                                                      v1v1_cost=6.51187283,
                                                      v2v2_cost=9.35785251,
                                                      p1v1_cost=0.,
                                                      p1v2_cost=0.,
                                                      p2v1_cost=0.,
                                                      p2v2_cost=0.,
                                                      u1u1_cost=1.02354949,
                                                      u2u2_cost=1.02354949,
                                                      u1u2_cost=0.)

else:  # without lqr
    controller = SymbolicPFLController(mass,
                                       length,
                                       com,
                                       damping,
                                       gravity,
                                       cfric,
                                       inertia,
                                       torque_limit,
                                       robot,
                                       pfl_method)

sim = Simulator(plant=double_pendulum)

controller.set_goal([np.pi, 0, 0, 0])

controller.set_cost_parameters_(par)

print(f"Simulating {pfl_method} PFL controller for {robot}")
print(f"Coulomb Friction:  {with_cfric}")
print(f"LQR: {with_lqr}")
print(f"dt: {dt}")
print(f"t final: {t_final}")
print(f"Cost parameters: {par}")

controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=x0,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator="runge_kutta",
                                   phase_plot=False,
                                   save_video=False)

# controller.save(path)
energy = controller.en
des_energy = controller.desired_energy

plot_timeseries(T, X, U, energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                energy_y_lines=[des_energy])

# Save Trajectory to a csv file to be sent to the motor.

# csv_data = np.vstack((T, np.asarray(X).T[0], np.asarray(X).T[1], U)).T
# np.savetxt("traj_opt_traj.csv",
#              csv_data,
#              delimiter=',',
#              header="time,pos,vel,torque",
#              comments="")
