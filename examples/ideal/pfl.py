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

pfl_method = "collocated"
with_lqr = True

if robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# simulation parameters
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]
dt = 0.01
x0 = [0.1, 0.0, 0.0, 0.0]
t_final = 10.0

# controller parameters
if robot == "acrobot":
    # lqr parameters
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
    if pfl_method == "collocated":
        #par = [6.78389278, 5.66430937, 9.98022384]
        #par = [1.58316202e-03, 2.94951787e+00, 1.44919303e+00]
        #par = [19.95373044, 14.76768604, 18.23010249]
        #par = [9.83825279, 9.42196979, 7.56036347]
        par = [0.0093613, 0.99787652, 0.9778557 ]  # new2
        #par = [4.95841985, 6.15434537, 9.54086796]  # new2
    elif pfl_method == "noncollocated":
        par = [9.19534629, 2.24529733, 5.90567362]
elif robot == "pendubot":
    # lqr parameters
    Q = np.diag([0.00125, 0.65, 0.000688, 0.000936])
    R = np.diag([25.0, 25.0])
    if pfl_method == "collocated":
        #par = [8.0722899, 4.92133648, 3.53211381]
        par = [8.8295605, 6.78718988, 4.42965278]
        #par = [5.23776024, 4.87113077, 3.0001595]
    elif pfl_method == "noncollocated":
        #par = [26.34039456, 99.99876263, 11.89097532]
        par = [8.0722899, 4.92133648, 3.53211381]

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
    # controller.lqr_controller.set_parameters(failure_value=np.nan,
    #                                          cost_to_go_cut=1000)
else:  # without lqr
    controller = SymbolicPFLController(model_pars=mpar,
                                       robot=robot,
                                       pfl_method=pfl_method)

sim = Simulator(plant=plant)

controller.set_goal(goal)
controller.set_cost_parameters_(par)

print(f"Simulating {pfl_method} PFL controller for {robot}")
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

save_trajectory(csv_path=os.path.join(save_dir, "trajectory.csv"),
                T=T,
                X=X,
                U=U)
mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
controller.save(save_dir)

plot_timeseries(T=T, X=X, U=U, energy=energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                energy_y_lines=[des_energy],
                save_to=os.path.join(save_dir, "time_series"))

