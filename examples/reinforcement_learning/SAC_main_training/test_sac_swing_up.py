import os

import numpy as np
from controller import SACController
from stable_baselines3 import SAC

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries

# hyperparameters
robot = "acrobot"

design = "design_C.1"
model = "model_1.0"
max_torque = 3.5
scaling_state = True

if robot == "pendubot":
    torque_limit = [max_torque, 0.0]
    active_act = 0

elif robot == "acrobot":
    torque_limit = [0.0, max_torque]
    # model_path = "models/max_torque_1.5_robustness_0.2"
    model_path = "acrobot_score_0500"
    #model_path = "/home/alberto_sinigaglia/double_pendulum/examples/reinforcement_learning/SAC_main_training/log_data/SAC_training/run_baseline_easy4.py-1.5-0.0-0-0/best_model/best_model.zip"

# import model parameter
model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.01
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
    scaling=scaling_state,
    max_velocity=50,
    torque_limit=torque_limit,
)

model = SAC.load(model_path)

# initialize sac controller
controller = SACController(model, dynamics_func, include_time=False)
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
)


# plot time series
# plot_timeseries(
#     T,
#     X,
#     U,
#     X_meas=sim.meas_x_values,
#     pos_y_lines=[np.pi],
#     tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
#     save_to=os.path.join("./", "timeseries"),
#     show=False,
# )
