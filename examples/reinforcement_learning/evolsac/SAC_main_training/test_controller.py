import os

import gymnasium as gym
import numpy as np
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from stable_baselines3 import SAC

robot = "pendubot"
max_velocity = 50.0
max_torque = 3.0
active_act = 0 if robot == "pendubot" else 1

torque_limit = [max_torque, 0] if robot == "pendubot" else [0, max_torque]

design = "design_C.1"
model = "model_1.0"
model_par_path = f"../../../../data/system_identification/identified_parameters/{design}/{model}/model_parameters.yml"

model_path = "./6.zip"


mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.01
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

state_representation = 2
# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
sac_model = SAC.load(
    model_path,
    custom_objects={"observation_space": obs_space, "action_space": act_space},
)

# initialize sac controller
controller = EvolSACController(
    model=sac_model,
    dynamics_func=dynamics_func,
    include_time=False,
    window_size=0,
)

# initialize lqr controller
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

# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    save_to="./timeseries.png",
)
