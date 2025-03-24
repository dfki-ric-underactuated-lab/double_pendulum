import os
from datetime import datetime
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator

from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

from double_pendulum.controller.acados_mpc.acados_mpc import AcadosMpc

actuated_joint = 0
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
gravity = 9.81
inertia = [0.09, 0.09]
actuated_joint=0    #0 = pendubot, 1 = acrobot
gravity=9.81
cfric = [0.093, 0.186]
Fmax = 6    #torque limit
bend_the_rules=True

torque_limit = [0.0, 0.0]
torque_limit[actuated_joint] = Fmax
if bend_the_rules:
    torque_limit[np.argmin(torque_limit)] = 0.5

# controller parameters
N_horizon=20
prediction_horizon=0.5
Nlp_max_iter=100000000

if actuated_joint == 1: #acrobot
    Q_mat = 2*np.diag([1000, 1000, 100, 100])
    Qf_mat = 2*np.diag([100000, 100000, 100, 100])
    R_mat = 2*np.diag([0.0001, 0.0001])

if actuated_joint == 0: #pendubot
    Q_mat = 2*np.diag([1000, 1000, 100, 100])
    Qf_mat = 2*np.diag([100000, 100000, 100, 100]) 
    R_mat = 2*np.diag([0.0001, 0.0001])

vmax = 30 #rad/s
vf = 0 # terminal velcoity constraint

mpc_cycle_dt = 0.005 # inner loop
outer_cycle_dt = 0.005 # outer loop

max_solve_time = mpc_cycle_dt * 20

bend_the_rules = True
friction_compensation_mpc = True
friction_compensation_added = False
solver_type = "SQP_RTI" #Select here [SQP, SQP_RTI, or DDP]

# simulation parameter
dt = outer_cycle_dt
t_final = 20  # 5.985
start = np.array([0,0,0,0])
goal = np.array([np.pi, 0, 0.0, 0.0])

# simulation plant
plant = SymbolicDoublePendulum(
    mass,
    length , 
    com , 
    damping , 
    gravity, 
    cfric , 
    inertia , 
    torque_limit=torque_limit
)

sim = Simulator(plant=plant)

if not friction_compensation_mpc:
    cfric = np.zeros_like(cfric)

controller = AcadosMpc(
    mass,
    length , 
    com , 
    damping ,
    cfric ,
    gravity, 
    inertia , 
    torque_limit
)

controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(
    N_horizon=N_horizon,
    prediction_horizon=prediction_horizon,
    Nlp_max_iter=Nlp_max_iter,
    max_solve_time=max_solve_time,
    solver_type=solver_type,
    wrap_angle=False,
    fallback_on_solver_fail=True,
    pd_KP=0.0,
    pd_KD=0.0,
    pd_KI=0,
    cheating_on_inactive_joint=bend_the_rules,
    mpc_cycle_dt=mpc_cycle_dt,
    outer_cycle_dt=outer_cycle_dt,
    pd_tracking=False,
    warm_start=True
)

controller.set_velocity_constraints(v_max=vmax, v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)
if friction_compensation_added:
    controller.set_friction_compensation(coulomb_fric=cfric)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_forecast=True,
                                   scale=0.5,
                                   save_video=False)

plot_timeseries(T, X, U, None,
               plot_energy=False,
               pos_y_lines=[0.0, np.pi],
               tau_y_lines=[-torque_limit[0], torque_limit[0],torque_limit[1],torque_limit[1]],
               scale=0.5)