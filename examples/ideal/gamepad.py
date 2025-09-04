import os
from datetime import datetime
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.gamepad.gamepad_pid_controller import (
    GamepadPIDController,
)
from double_pendulum.utils.plotting import plot_timeseries


# model parameters
design = "design_C.1"
model = "model_1.1"
robot = "double_pendulum"

torque_limit_simulation = [10.0, 10.0]
torque_limit_controller = [10.0, 10.0]

model_par_path = (
    "../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)

mpar = model_parameters()
mpar.load_yaml(model_par_path)
# mpar.set_motor_inertia(0.)
# mpar.set_cfric([0., 0.])
# mpar.set_damping([0., 0.])
mpar.set_torque_limit(torque_limit_simulation)

# simulation parameters
dt = 0.01
t_final = 30.0
integrator = "runge_kutta"
x0 = [0.0, 0.0, 0.0, 0.0]

plant = DoublePendulumPlant(model_pars=mpar)
sim = Simulator(plant=plant)

controller = GamepadPIDController(
    torque_limit=torque_limit_controller,
    pid_gains=[2.0, 0.0, 0.5],
    gamepad_name="Logitech Gamepad F710",
    max_vel=5.0,
)
controller.set_gravity_compensation(plant=plant)
controller.use_gravity_compensation = True
# controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
controller.init()

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=x0,
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
    pos_y_lines=[0.0, np.pi],
    tau_y_lines=[-torque_limit_controller[0], torque_limit_controller[0]],
)
