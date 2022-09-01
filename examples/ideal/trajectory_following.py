import os
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController, TrajectoryInterpController
from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties

design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [6.0, 0.0]
if robot == "double_pendulum":
    torque_limit = [6.0, 6.0]

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# csv file
use_feed_forward_torque = True
csv_path = os.path.join("../../data/trajectories/",
                        design, traj_model, robot,
                        "ilqr_1/trajectory.csv")

T_des, X_des, U_des = load_trajectory(csv_path)
dt, t_final, x0, _ = trajectory_properties(T_des, X_des)
goal = [np.pi, 0., 0., 0.]
integrator = "runge_kutta"

## plant simulator and controller
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TrajectoryController(csv_path=csv_path,
                                  torque_limit=torque_limit,
                                  kK_stabilization=True)
# controller = TrajectoryInterpController(csv_path=csv_path,
#                                   torque_limit=torque_limit,
#                                   kK_stabilization=True,
#                                   num_break=40)
#controller = TrajPIDController(csv_path=csv_path,
#                           use_feed_forward_torque=use_feed_forward_torque,
#                           torque_limit=torque_limit)
#controller.set_parameters(Kp=200.0, Ki=0.0, Kd=2.0)

controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator, phase_plot=False,
                                   plot_inittraj=True, plot_forecast=False,
                                   save_video=False, video_name="acrobot_swingup")
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                T_des=T_des, X_des=X_des, U_des=U_des)
