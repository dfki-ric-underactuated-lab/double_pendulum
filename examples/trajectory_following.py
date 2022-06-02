import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController, TrajectoryInterpController
from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties

robot = "acrobot"
trajopt = "ilqr"

cfric = [0., 0.]
motor_inertia = 0.
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]

# model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
model_par_path = "../data/system_identification/identified_parameters/tmotors_v2.0/model_parameters_est.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

# csv file
use_feed_forward_torque = True
read_with = "numpy"
keys = ""
#latest_dir = sorted(os.listdir(os.path.join("data", robot, trajopt, "trajopt")))[-1]
#csv_path = os.path.join("data", robot, trajopt, "trajopt", latest_dir, "trajectory.csv")
#csv_path = "../data/trajectories/acrobot/ilqr_v1.0/trajectory.csv"
csv_path = "../data/trajectories/acrobot/ilqr/trajectory.csv"

T_des, X_des, U_des = load_trajectory(csv_path, read_with=read_with)
dt, t_final, x0, _ = trajectory_properties(T_des, X_des)

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=csv_path,
                                  read_with=read_with,
                                  keys=keys,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)
# controller = TrajectoryInterpController(csv_path=csv_path,
#                                   torque_limit=torque_limit,
#                                   read_with=read_with,
#                                   keys=keys,
#                                   kK_stabilization=True,
#                                   num_break=40)
#controller = TrajPIDController(csv_path=csv_path,
#                           read_with=read_with,
#                           use_feed_forward_torque=use_feed_forward_torque,
#                           torque_limit=torque_limit)
#controller.set_parameters(Kp=200.0, Ki=0.0, Kd=2.0)
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False,
                                   plot_inittraj=True, plot_forecast=False)
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                T_des=T_des, X_des=X_des, U_des=U_des)
