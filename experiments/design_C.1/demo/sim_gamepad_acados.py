import os
from datetime import datetime
import numpy as np

from double_pendulum.model.plant import DoublePendulumPlant
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.controller.gamepad.gamepad import GamePad
from double_pendulum.controller.combined_controller import CombinedControllerN

from con_gamepad import controller as con_gamepad
from con_safety import controller as con_safety
from con_acados_mpc_acrobot import controller as con_acrobot
from con_acados_mpc_pendubot import controller as con_pendubot

from parameters import dt, t_final, goal, x0, model_par_path, torque_limit, integrator

GP = GamePad("Logitech Gamepad F710")

mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_torque_limit(torque_limit)

plant = DoublePendulumPlant(model_pars=mpar)
sim = Simulator(plant=plant)


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
)
