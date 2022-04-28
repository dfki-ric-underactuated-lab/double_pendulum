import os
from datetime import datetime
import numpy as np

from double_pendulum.trajectory_optimization.direct_collocation.direct_collocation import dircol_calculator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
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
torque_limit = 2.0
theta_limit = float(np.deg2rad(360.))
speed_limit = 7
minimum_timestep = 0.05
maximum_timestep = 0.4

# costs
R = 2
time_panalization = 0

# Direct Collocation calculation
dc = dircol_calculator(urdf_path, robot)
dc.compute_trajectory(
    n=n,
    tau_limit=torque_limit,
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
                tau_y_lines=[-torque_limit, torque_limit],
                save_to=os.path.join(save_dir, "timeseries"))

# animate
# animation with meshcat in browser window
# dc.animate_trajectory()

# simulate in python plant
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
#damping = [0.0, 0.0]
cfric = [0.0, 0.0]
gravity = 9.81
inertia = [0.05472, 0.02522]
torque_limit = [0.0, 3.0]

dt = T[1] - T[0]
t_final = T[-1]
x0 = X[0]

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=traj_file,
                                  torque_limit=torque_limit,
                                  kK_stabilization=False)

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=True)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]])
