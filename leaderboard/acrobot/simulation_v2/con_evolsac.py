import stable_baselines3
from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.simulation import Simulator
from sim_parameters import design, integrator, model, mpar, robot
from stable_baselines3 import SAC

assert stable_baselines3.__version__ == "2.3.2"

name = "evolsac"
leaderboard_config = {
    "csv_path": name + "/sim_swingup.csv",
    "name": name,
    "simple_name": "evolsac",
    "short_description": "Evolutionary SAC for both swingup and stabilisation",
    "readme_path": f"readmes/{name}.md",
    "username": "AlbertoSinigaglia",
}

# All of the variables below are now imported from sim_parameters.py
# robot = "acrobot"
# design = "design_C.1"
# model = "model_1.0"

# integrator = "runge_kutta"

max_torque = 3.0
max_velocity = 50.0
torque_limit = [max_torque, 0.0] if robot == "pendubot" else [0.0, max_torque]
mpar.set_torque_limit(torque_limit)

print(f"Loading {robot} trained model...")

model_path = "../../../data/policies/design_C.1/model_1.0/acrobot/evolsac/model.zip"
# Simulation parameters
dt = 0.01

# learning environment parameters
state_representation = 2

plant = SymbolicDoublePendulum(model_pars=mpar)
print("Loading model parameters...")
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

sac_model = SAC.load(model_path)
controller = EvolSACController(
    model=sac_model, dynamics_func=dynamics_func, window_size=0, include_time=False
)
controller.init()
