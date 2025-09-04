from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.pid.point_pid_controller import (
    PointPIDController,
)
from parameters import model_par_path

torque_limit_controller = [2.0, 2.0]

mpar = model_parameters()
mpar.load_yaml(model_par_path)

plant = DoublePendulumPlant(model_pars=mpar)

controller = PointPIDController(
    torque_limit=torque_limit_controller,
    dt=0.003,
    modulo_angles=False,
    pos_contribution_limit=[1.0, 1.0],
)
controller.set_parameters(2.0, 0.0, 0.3)
controller.set_goal([0.0, 0.0, 0.0, 0.0])

controller.set_gravity_compensation(plant=plant)
controller.use_gravity_compensation = True
controller.init()
