import os
import torch
import gymnasium as gym
import numpy as np
from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.perturbations import get_random_gauss_perturbation_array
from double_pendulum.simulation.simulation import Simulator
from stable_baselines3 import SAC


class ExtendedEvolSACController(EvolSACController):
    def __init__(
        self,
        model,
        dynamics_func,
        scaling=True,
        include_time=True,
        window_size=0,
        SNES_phase=False,
        evaluating=True,
        ctrl_rate=1,
        wait_steps=0,
    ):
        # Call the parent class constructor
        super().__init__(
            model=model,
            dynamics_func=dynamics_func,
            scaling=scaling,
            include_time=include_time,
            window_size=window_size,
            SNES_phase=SNES_phase,
            evaluating=evaluating,
        )

        # Additional attributes for the extended class
        self.ctrl_rate = ctrl_rate
        self.ctrl_cnt = 0
        self.last_control = np.zeros(2)  # Assuming action space has 2 elements
        self.wait_steps = wait_steps

    def get_control_output_(self, x, t=None):
        """Override the control output to incorporate ctrl_rate and wait_steps logic."""
        self.timestep += 1

        # Control logic based on ctrl_rate and wait_steps
        if self.ctrl_cnt % self.ctrl_rate == 0 and self.ctrl_cnt >= self.wait_steps:
            obs = self.dynamics_func.normalize_state(x)
            self.update_old_state(obs, t)
            action = self.model.predict(self.get_state(obs, t), deterministic=True)

            # Add Gaussian noise if not evaluating and during SNES_phase
            if not self.evaluating and self.SNES_phase:
                a = action[0][0]
                a = torch.atanh(torch.tensor(a))
                a += np.random.randn() * 0.1
                a = torch.tanh(a).numpy()
                action[0][0] = np.clip(a, a_min=-1, a_max=+1)

            # Store the last control action
            self.last_control = action
            self.update_old_action(action)

        # Increment the control count
        self.ctrl_cnt += 1

        # Return the last control action, scaled back by the dynamics function
        return self.dynamics_func.unscale_action(self.last_control)


# Set random seed for the 10 experiments
# np.random.seed(591)

robot = "pendubot"
friction_compensation = False

# New models have all max_torque = 3.0, max_velocity = 50.0, window_size = 0
max_torque = 3.0
max_velocity = 50.0
window_size = 0
include_time = False
torque_limit = [0.0, max_torque] if robot == "acrobot" else [max_torque, 0.0]

# Time parameters
dt = 1 / 500
t_final = 10.0
# N = int(t_final / dt)
# T_des = np.linspace(0, t_final, N + 1)
# U_des = np.zeros((N + 1, 2))

# Load model parameters and setup plant
model_par_path = "../../data/system_identification/identified_parameters/design_C.1/model_1.0/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)
plant = SymbolicDoublePendulum(model_pars=mpar)

# Simulation setup
simulator = Simulator(plant=plant)
integrator = "runge_kutta"
state_representation = 2

dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

# # Controller frequency
control_frequency = 1 / 100
ctrl_rate = int(control_frequency / dt)

# model path entirely defined by terminal args
model_path = (
    f"../../data/policies/design_C.1/model_1.0/pendubot/evolsac/real_robot_model.zip"
)

# Load SAC model
obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
sac_model = SAC.load(
    model_path,
    custom_objects={"observation_space": obs_space, "action_space": act_space},
)


# SAC controller
controller = ExtendedEvolSACController(
    model=sac_model,
    dynamics_func=dynamics_func,
    window_size=window_size,
    include_time=include_time,
    ctrl_rate=ctrl_rate,
    wait_steps=0,
)

# Friction compensation
if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)

# Low-pass filter
filter = lowpass_filter(
    alpha=[1.0, 1.0, 0.2, 0.2], x0=[0.0, 0.0, 0.0, 0.0], filt_velocity_cut=0.1
)

controller.set_filter(filter)
controller.init()

# Perturbations
perturbation_array, _, _, _ = get_random_gauss_perturbation_array(
    t_final, dt, 2, 1.0, [0.05, 0.1], [0.2, 0.5]
)

# Run experiment
run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[3, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=[6.0, 0.5] if robot == "pendubot" else [0.5, 6.0],
    save_dir=os.path.join("data", robot, "evolsac_robust"),
    record_video=True,
    safety_velocity_limit=30.0,
    perturbation_array=perturbation_array,  # comment this line for perturbation-free tests
)
