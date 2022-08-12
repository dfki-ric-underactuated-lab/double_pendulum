import time
from datetime import datetime
import os
import numpy as np

from double_pendulum.trajectory_optimization.direct_collocation.direct_collocation import dircol_calculator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController


# model parameters
robot = "acrobot"
urdf_path = "../data/urdfs/"+robot+".urdf"

# Trajectory parameters
initial_state = (0.0, 0.0, 0., 0.)
final_state = (np.pi, 0.0, 0.0, 0.0)
n = 100
init_traj_time_interval = [0., 6.]
freq = 1000

# limits
torque_limit_active = 6.0
theta_limit = float(np.deg2rad(360.))
speed_limit = 7
minimum_timestep = 0.05
maximum_timestep = 0.4

# costs
R = 0.1
time_panalization = 0

# Direct Collocation calculation
t0 = time.time()
dc = dircol_calculator(urdf_path, robot)
dc.compute_trajectory(
    n=n,
    tau_limit=torque_limit_active,
    initial_state=initial_state,
    final_state=final_state,
    theta_limit=theta_limit,
    speed_limit=speed_limit,
    R=R,
    time_panalization=time_panalization,
    init_traj_time_interval=init_traj_time_interval,
    minimum_timestep=minimum_timestep,
    maximum_timestep=maximum_timestep)

T, X, U = dc.get_trajectory(freq=freq)
print("Computing time: ", time.time() - t0, "s")

# saving
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "dircol", "trajopt", timestamp)
os.makedirs(save_dir)

traj_file = os.path.join(save_dir, "trajectory.csv")
save_trajectory(csv_path=traj_file,
                T=T, X=X, U=U)
# plotting
U = np.append(U, [[0.0, 0.0]], axis=0)
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit_active, torque_limit_active],
                save_to=os.path.join(save_dir, "timeseries"))

# animate
# animation with meshcat in browser window
# dc.animate_trajectory()

# simulate in python plant
cfric = [0., 0.]
# gravity = 9.81
# inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
motor_inertia = 0.
if robot == "acrobot":
    torque_limit = [0.0, torque_limit_active]
if robot == "pendubot":
    torque_limit = [torque_limit_active, 0.0]
if robot == "double_pendulum":
    torque_limit = [torque_limit_active, torque_limit_active]
model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
# mpar.set_damping(damping)
mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

dt = T[1] - T[0]
t_final = T[-1]
x0 = X[0]

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=traj_file,
                                  read_with="numpy",
                                  keys="",
                                  torque_limit=torque_limit,
                                  kK_stabilization=False)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=True)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit_active, torque_limit_active])
