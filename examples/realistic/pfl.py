import sys
import os
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import (SymbolicPFLController,
                                                                                    SymbolicPFLAndLQRController)

# model parameters
design = "design_A.0"
model = "model_2.0"
robot = "acrobot"
with_cfric = False

motor_inertia = 0.
#damping = [0.081, 0.0]
#damping = [0.0, 0.0]
if not with_cfric:
    cfric = [0.0, 0.0]
gravity = 9.81
if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
# mpar.set_damping(damping)
if not with_cfric:
    mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

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
            #par = [9.94271982, 1.56306923, 3.27636175]  # ok
            par = [6.97474837, 9.84031538, 9.1297417]  # v1.0
        else:
            #par = [9.98906556, 5.40486824, 7.28776292]  # similar to the one below
            #par = [7.5, 4.4, 7.3]  # best
            #par = [0.5, 0.44, 0.6]
            #par = [9.39406094, 8.45818256, 0.82061105]
            #par = [14.07071258, 4.31253548, 16.77354333] # quick start
            par = [6.78389278, 5.66430937, 9.98022384]
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


plant = SymbolicDoublePendulum(model_pars=mpar)

if with_lqr:
    controller = SymbolicPFLAndLQRController(model_pars=mpar,
                                             robot=robot,
                                             pfl_method=pfl_method)
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
    controller = SymbolicPFLController(model_pars=mpar,
                                       robot=robot,
                                       pfl_method=pfl_method)

sim = Simulator(plant=plant)

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

energy = controller.en
des_energy = controller.desired_energy

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "pfl", pfl_method, timestamp)
os.makedirs(save_dir)

# controller.save(save_dir)

save_trajectory(csv_path=os.path.join(save_dir, "trajectory.csv"),
                T=T,
                X=X,
                U=U)

plot_timeseries(T=T, X=X, U=U, energy=energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                energy_y_lines=[des_energy],
                save_to=os.path.join(save_dir, "time_series"))

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
controller.save(save_dir)
