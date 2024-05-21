import os
import numpy as np

from double_pendulum.controller.trajectory_following.feed_forward import (
    FeedForwardController,
)
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment

# from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
# from double_pendulum.model.model_parameters import model_parameters
# from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.filter.identity import identity_filter


design = "design_C.0"
torque_limit = [0.0, 0.0]

# trajectory
dt = 0.005
t_final = 10.0
N = int(t_final / dt)
T_des = np.linspace(0, t_final, N + 1)
u1 = np.zeros(N + 1)
u2 = np.zeros(N + 1)
U_des = np.array([u1, u2]).T

# measurement filter
# lowpass_alpha = [1.0, 1.0, 0.2, 0.2]
filter_velocity_cut = 0.0

# filter
filter = identity_filter(filter_velocity_cut)

# controller
controller = FeedForwardController(
    T=T_des, U=U_des, torque_limit=[0.0, 0.0], num_break=40
)
controller.set_filter(filter)


# gravity and friction compensation
# model_par_path = "../data/system_identification/identified_parameters/design_A.0/model_1.0/model_parameters.yml"
# mpar = model_parameters(filepath=model_par_path)
# plant = SymbolicDoublePendulum(model_pars=mpar)
# controller.set_gravity_compensation(plant=plant)

# controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
# controller.set_friction_compensation(damping=[0.005, 0.001], coulomb_fric=[0.093, 0.15])
# controller.set_friction_compensation(damping=[0.0, 0.01], coulomb_fric=[0.08, 0.04])
# controller.set_friction_compensation(damping=[0.001, 0.001], coulomb_fric=[0.09, 0.078])
# controller.set_friction_compensation(damping=[0.001, 0.001], coulomb_fric=[0.12, 0.078])

controller.init()

run_experiment(
    controller=controller,
    dt=dt,
    t_final=t_final,
    can_port="can0",
    motor_ids=[7, 8],
    tau_limit=torque_limit,
    save_dir=os.path.join("data", design, "double-pendulum/tmotors/donothing"),
)

