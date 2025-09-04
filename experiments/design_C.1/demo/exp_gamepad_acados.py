import os
import numpy as np

from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.controller.gamepad.gamepad import GamePad
from double_pendulum.controller.combined_controller import CombinedControllerN

from parameters import dt, t_final, torque_limit
from con_gamepad import controller as con_gamepad
from con_safety import controller as con_safety
from con_acados_mpc_acrobot import controller as con_acrobot
from con_acados_mpc_pendubot import controller as con_pendubot

GP = GamePad("Logitech Gamepad F710")


def cond_switch_to_gamepad(x, t):
    inp = GP.read()
    return inp[4]  # start button


def cond_switch_to_acados_pendu(x, t):
    inp = GP.read()
    return inp[6]  # left bumper


def cond_switch_to_acados_acro(x, t):
    inp = GP.read()
    return inp[7]  # right bumper


def cond_switch_to_safety(x, t):
    inp = GP.read()
    switch = inp[5]  # back button
    if np.max(np.abs(x[:2])) > 3.5 * np.pi:
        switch = True
    if np.max(np.abs(x[2:])) > 20.0:
        switch = True
    return switch


controller = CombinedControllerN(
    n_con=4,
    controllers=[con_gamepad, con_pendubot, con_acrobot, con_safety],
    conditions=[
        cond_switch_to_gamepad,
        cond_switch_to_acados_pendu,
        cond_switch_to_acados_acro,
        cond_switch_to_safety,
    ],
    compute_in_bg=[False, True, True, False],
    verbose=True,
)

data_dir = "data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
save_dir = os.path.join(data_dir, "gamepad_runs")

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[2, 1],
    motor_directions=[1.0, -1.0],
    tau_limit=torque_limit,
    save_dir=save_dir,
    record_video=False,
    safety_velocity_limit=25.0,
    safety_position_limit=4 * np.pi,
)
