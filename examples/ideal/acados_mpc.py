import os
from datetime import datetime
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator

from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

from double_pendulum.controller.acados_mpc_acados_mpc_controller import AcadosMpcController

## labels
design = "design_C.0"
model = "model_3.0"
traj_model = "model_3.1"

# model parameter
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
gravity = 9.81
inertia = [0.09, 0.09]
Fmax = 6    #torque limit
actuated_joint=0    #0 = pendubot, 1 = acrobot
gravity=9.81
bend_the_rules=True
cfric = [0.093, 0.186]

torque_limit = [0.0, 0.0]
torque_limit[actuated_joint] = Fmax
if bend_the_rules:
    torque_limit[np.argmin(torque_limit)] = 0.5

# simulation parameter
dt = 0.01
t_final = 10  # 5.985
start = np.array([np.pi/6, 0, 0.0, 0.0])
goal = np.array([np.pi, 0, 0.0, 0.0])

# controller parameters
N_horizon=20
prediction_horizon=2.0
Nlp_max_iter=200

if actuated_joint == 1: #acrobot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([10000, 10000, 100, 100])
    R_mat = 2*np.diag([0.0001, 0.0001])

if actuated_joint == 0: #pendubot
    Q_mat = 2*np.diag([100, 100, 10, 10])
    Qf_mat = 2*np.diag([10000, 10000, 100, 100]) 
    R_mat = 2*np.diag([0.0001, 0.0001])

vmax = 30 #rad/s
vf = 0

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
robot = "pendubot" if actuated_joint == 0 else "acrobot"
save_dir = os.path.join("data", design, model, robot, "ilqr", "mpc", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(
    mass=mass,
    length=length,
    com=com,
    damping=damping,
    gravity=gravity,
    coulomb_fric=cfric,
    inertia=inertia,
    torque_limit=torque_limit
)

sim = Simulator(plant=plant)

controller = AcadosMpcController(
    mass=mass,
    length=length, 
    damping=damping, 
    coulomb_fric=cfric,
    inertia=inertia,
    com=com,
    gravity=gravity,
    torque_limit=torque_limit,
    cheating_on_inactive_joint= np.min(torque_limit)
)

controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(
    N_horizon=N_horizon,
    prediction_horizon=prediction_horizon,
    Nlp_max_iter=Nlp_max_iter,
    solver_type="SQP",
    wrap_angle=False
)

#controller.set_velocity_constraints(v_max=vmax)
#controller.set_velocity_constraints(v_final=vf)
controller.set_cost_parameters(Q_mat=Q_mat, Qf_mat=Qf_mat, R_mat=R_mat)
#controller.load_init_traj(csv_path=init_csv_path)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_forecast=True,
                                   scale=0.5)

#plot_timeseries(T, X, U, None,
 #               plot_energy=False,
  #              pos_y_lines=[0.0, np.pi],
   #             tau_y_lines=[-torque_limit[1], torque_limit[1]],
 #               scale=0.5)
#
controller.save(save_dir)