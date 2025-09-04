from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.gamepad.gamepad_pid_controller import (
    GamepadPIDController,
)
from parameters import model_par_path

torque_limit_controller = [5.0, 5.0]

mpar = model_parameters()
mpar.load_yaml(model_par_path)

plant = DoublePendulumPlant(model_pars=mpar)

controller = GamepadPIDController(
    torque_limit=torque_limit_controller,
    pid_gains=[2.0, 0.0, 0.3],
    gamepad_name="Logitech Gamepad F710",
    max_vel=5.0,
)
controller.set_gravity_compensation(plant=plant)
controller.use_gravity_compensation = True
controller.init()
