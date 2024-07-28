import os
import sys

from stable_baselines3 import SAC

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.simulation import Simulator

class SACController(AbstractController):
    def __init__(
        self, model, dynamics_func, scaling=True, include_time=True, window_size=0
    ):
        super().__init__()
        self.model = model
        self.dynamics_func = dynamics_func
        self.scaling = scaling
        self.window_size = window_size
        self.include_time = include_time
        self.old_state = [
            [0] * (5 if include_time else 4) for _ in range(self.window_size)
        ]
        self.old_action = [[0.0] for _ in range(self.window_size)]
        self.timestep = 0

    def get_state(self, obs, time):
        if self.window_size > 0:
            return np.concatenate(
                [np.reshape(self.old_state, (-1)), np.reshape(self.old_action, (-1))]
            )
        else:
            if self.include_time:
                return list(obs) + [time / 10]
            else:
                return obs

    def update_old_state(self, obs, t):
        if self.include_time:
            self.old_state = self.old_state[1:] + [list(obs) + [t / 10]]
        else:
            self.old_state = self.old_state[1:] + [list(obs)]

    def update_old_action(self, action):
        self.old_action = self.old_action[1:] + [action[0]]

    def get_control_output_(self, x, t=None):
        self.timestep += 1
        obs = self.dynamics_func.normalize_state(x)
        self.update_old_state(obs, t)
        action = self.model.predict(self.get_state(obs, t), deterministic=True)
        self.update_old_action(action)
        return self.dynamics_func.unscale_action(action)


name = "sac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "sac",
    "short_description": "SAC for both swingup and stabilisation",
    "readme_path": f"readmes/{name}.md",
    "username": "MarcoCali0",
}


robot = "pendubot"
print("Loading trained model...")

max_velocity = 50.0

active_act = 0 if robot == "pendubot" else 1

design = "design_C.1"
model = "model_1.0"

model_par_path = (
    "/home/alberto_sinigaglia/double_pendulum/data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)

max_torque = 3.0
torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]

model_path = "../model.zip"

# Model parameters
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

print("Loading model parameters...")

# Simulation parameters
dt = 0.01

# learning environment parameters
state_representation = 2

integrator = "runge_kutta"
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

model = SAC.load(model_path)
controller = SACController(
    model=model, dynamics_func=dynamics_func, window_size=0, include_time=False
)
controller.init()
