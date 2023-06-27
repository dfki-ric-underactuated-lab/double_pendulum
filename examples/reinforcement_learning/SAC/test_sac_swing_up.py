import os
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.simulation.gym_env import (
    double_pendulum_dynamics_func,
)


class SACController(AbstractController):
    def __init__(self, model_path, dynamics_func, dt):
        super().__init__()

        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action = self.model.predict(obs)
        u = self.dynamics_func.unscale_action(action)
        return u


# model parameters
design = "design_C.0"
model = "model_3.0"
# robot = "pendubot"
robot = "acrobot"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

model_par_path = (
    "/home/chi/Github/double_pendulum/data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.01
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=2,
)

controller = SACController(
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/best_model/acrobot_model.zip",
    # pendubot
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v0_5e6/acrobot_model.zip",
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v1_1e6/acrobot_model.zip",
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v2/acrobot_model.zip",
    # model_path="/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/pendubot_2e6/v3/acrobot_model.zip",

    # acrobot
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v0_5e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v1_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v2_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v3_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v4_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v5_1e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_2e6/v6_1e6/acrobot_model.zip",

    # acrobot quadratic modifeid reward
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v0/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v2/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v3/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v4_with_termination/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_1e6_quadraticModified_reward/v5_without_termination_5e7/acrobot_model.zip",

    # acrobot with speed limitation
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_speed_modified/v0_5e6/acrobot_model.zip",
    # model_path = "/home/chi/Github/double_pendulum/src/python/double_pendulum/controller/SAC/sac_training/acrobot_speed_modified/v1_5e6/acrobot_model.zip",
    model_path ="sac_training/acrobot/acrobot_speed_modified/v2_5e6_final/best_model.zip",
    dynamics_func=dynamics_func,
    dt=dt,
)
controller.init()

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
)
