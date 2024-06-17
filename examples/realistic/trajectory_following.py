import os
import sys
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import (
    TrajectoryController,
    TrajectoryInterpController,
)
from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties
from double_pendulum.filter.lowpass import lowpass_filter

PLOT = "plot" in sys.argv
ANIMATE = "animate" in sys.argv

design = "design_C.0"
model = "model_3.0"
traj_model = "model_3.1"
robot = "acrobot"

friction_compensation = True

if robot == "acrobot":
    torque_limit = [0.0, 6.0]
elif robot == "pendubot":
    torque_limit = [6.0, 0.0]
else:
    torque_limit = [6.0, 6.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

# csv file
use_feed_forward_torque = True
csv_path = os.path.join(
    "../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv"
)

T_des, X_des, U_des = load_trajectory(csv_path)
dt, t_final, x0, _ = trajectory_properties(T_des, X_des)
goal = [np.pi, 0.0, 0.0, 0.0]
integrator = "runge_kutta"

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.1, 0.1]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0.0, 0.0]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
lowpass_alpha = [1.0, 1.0, 0.3, 0.3]
filter_velocity_cut = 0.1

## plant
plant = DoublePendulumPlant(model_pars=mpar)

# simulator
sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(
    meas_noise_sigmas=meas_noise_sigmas, delay=delay, delay_mode=delay_mode
)
sim.set_motor_parameters(
    u_noise_sigmas=u_noise_sigmas, u_responsiveness=u_responsiveness
)

# filter
filter = lowpass_filter(lowpass_alpha, x0, filter_velocity_cut)

# controller
# controller = TrajectoryController(csv_path=csv_path,
#                                   torque_limit=torque_limit,
#                                   kK_stabilization=True)
controller = TrajectoryInterpController(
    csv_path=csv_path, torque_limit=torque_limit, kK_stabilization=True, num_break=40
)
# controller = TrajPIDController(csv_path=csv_path,
#                           use_feed_forward_torque=use_feed_forward_torque,
#                           torque_limit=torque_limit)
# controller.set_parameters(Kp=200.0, Ki=0.0, Kd=2.0)

controller.set_filter(filter)

if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
controller.init()

if ANIMATE:
    T, X, U = sim.simulate_and_animate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
        phase_plot=False,
        plot_inittraj=True,
        plot_forecast=False,
    )
else:
    T, X, U = sim.simulate(
        t0=0.0,
        x0=x0,
        tf=t_final,
        dt=dt,
        controller=controller,
        integrator=integrator,
    )

if PLOT:
    plot_timeseries(
        T,
        X,
        U,
        None,
        plot_energy=False,
        pos_y_lines=[0.0, np.pi],
        T_des=T_des,
        X_des=X_des,
        U_des=U_des,
    )
