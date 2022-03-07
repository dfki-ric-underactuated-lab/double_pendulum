import sys
import os
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.saving import save_trajectory
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import (SymbolicPFLController,
                                                                                    SymbolicPFLAndLQRController)

# model parameters
robot = "acrobot"
with_cfric = False

mass = [0.608, 0.5]
length = [0.3, 0.4]
com = [length[0], length[1]]
#damping = [0.081, 0.0]
damping = [0.0, 0.0]
if with_cfric:
    cfric = [0.093, 0.186]
else:
    cfric = [0.0, 0.0]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

# simulation parameters
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]
dt = 0.01
x0 = [0.1, 0.0, 0.0, 0.0]
t_final = 10.0

# controller parameters
pfl_method = "collocated"
with_lqr = True

if robot == "acrobot":
    # lqr parameters
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
    if pfl_method == "collocated":
        if with_cfric:
            par = [9.94271982, 1.56306923, 3.27636175]  # ok
        else:
            #par = [9.98906556, 5.40486824, 7.28776292]  # similar to the one below
            par = [7.5, 4.4, 7.3]  # best
            #par = [0.5, 0.44, 0.6]
            #par = [9.39406094, 8.45818256, 0.82061105]
            #par = [14.07071258, 4.31253548, 16.77354333] # quick start
    elif pfl_method == "noncollocated":
        par = [9.19534629, 2.24529733, 5.90567362]  # good
    else:
        print(f"pfl_method {pfl_method} not found. Please set eigher" +
              "pfl_method='collocated' or pfl_method='noncollocated'")
        sys.exit()
elif robot == "pendubot":
    # lqr parameters
    Q = np.diag((11.64, 79.58, 0.073, 0.0003))
    R = np.diag((0.13, 0.13))
    if pfl_method == "collocated":
        par = [6.97474837, 9.84031538, 9.1297417]  # bad
    elif pfl_method == "noncollocated":
        par = [26.34039456, 99.99876263, 11.89097532]
    else:
        print(f"pfl_method {pfl_method} not found. Please set eigher" +
              "pfl_method='collocated' or pfl_method='noncollocated'")
        sys.exit()
else:
    print(f"robot {robot} not found. Please set eigher" +
          "robot='acrobot' or robot='pendubot'")
    sys.exit()


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
    controller.lqr_controller.set_cost_parameters(p1p1_cost=Q[0, 0],
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

controller.set_goal(goal)
controller.set_cost_parameters_(par)

print(f"Simulating {pfl_method} PFL controller for {robot}")
print(f"Coulomb Friction: {with_cfric}")
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
                                   integrator=integrator,
                                   phase_plot=False,
                                   save_video=False)

# controller.save(path)
energy = controller.en
des_energy = controller.desired_energy

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "pfl", pfl_method, timestamp)
os.makedirs(save_dir)

save_trajectory(filename=os.path.join(save_dir, "trajectory.csv"),
                T=T,
                X=X,
                U=U)

plot_timeseries(T, X, U, energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                energy_y_lines=[des_energy],
                save_to=os.path.join(save_dir, "time_series"))

par_dict = {"mass1": mass[0],
            "mass2": mass[1],
            "length1": length[0],
            "length2": length[1],
            "com1": com[0],
            "com2": com[1],
            "inertia1": inertia[0],
            "inertia2": inertia[1],
            "damping1": damping[0],
            "damping2": damping[1],
            "coulomb_friction1": cfric[0],
            "coulomb_friction2": cfric[1],
            "gravity": gravity,
            "torque_limit1": torque_limit[0],
            "torque_limit2": torque_limit[1],
            "dt": dt,
            "t_final": t_final,
            "integrator": integrator,
            "start_pos1": x0[0],
            "start_pos2": x0[1],
            "start_vel1": x0[2],
            "start_vel2": x0[3],
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3],
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
