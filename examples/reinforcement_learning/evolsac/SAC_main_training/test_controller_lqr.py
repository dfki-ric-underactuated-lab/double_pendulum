import copy
import os
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot
import numpy as np
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.controller.evolsac.evolsac_controller import EvolSACController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_diff, wrap_angles_top
from stable_baselines3 import SAC

robot = "pendubot"
max_velocity = 50.0
max_torque = 3.0
active_act = 0 if robot == "pendubot" else 1

torque_limit = [max_torque, 0] if robot == "pendubot" else [0, max_torque]

design = "design_C.1"
model = "model_1.0"
model_par_path = f"../../../../data/system_identification/identified_parameters/{design}/{model}/model_parameters.yml"

load_path = "../../../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr/"
scaling_state = True

model_path = "./pendubot_no_friction.zip"
model_path = "./6.zip"


mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"
goal = [np.pi, 0.0, 0.0, 0.0]

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

rho = np.loadtxt(os.path.join(load_path, "rho"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))
flag = False


def condition1(t, x):
    return False


# def condition2(t, x):
#     goal = [np.pi, 0.0, 0.0, 0.0]

#     delta = wrap_angles_diff(np.subtract(x, goal))

#     switch = False
#     if np.einsum("i,ij,j", delta, S, delta) < 1.0 * rho:
#         switch = True
#         print(f"Switch to LQR at time={t}")

#     return switch


def condition2(t, x):
    theta1, theta2, omega1, omega2 = x
    # theta1, theta2, omega1, omega2 = dynamics_func.unscale_state(x)
    link_end_points = dynamics_func.simulator.plant.forward_kinematics([theta1, theta2])

    y_ee = link_end_points[1][1]
    velocity_norm = np.sqrt(omega1**2 + omega2**2)

    if y_ee > 0.46 and velocity_norm < 0.03:
        print(f"time {t}, Height: {y_ee}. Velocity norm: {velocity_norm}")
        return True
    else:
        return False


state_representation = 2
# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=sim,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)

obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
sac_model = SAC.load(
    model_path,
    custom_objects={"observation_space": obs_space, "action_space": act_space},
)

# initialize sac controller
controller1 = EvolSACController(
    model=sac_model,
    dynamics_func=dynamics_func,
    include_time=False,
    window_size=0,
)

# initialize lqr controller
mpar = copy.deepcopy(mpar)
# mpar.set_motor_inertia(0.0)
# mpar.set_damping([0.0, 0.0])
# mpar.set_cfric([0.0, 0.0])

controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
# LQR parameters
lqr_pars = np.loadtxt(os.path.join(load_path, "controller_par.csv"))
Q = np.diag((0.97, 0.93, 0.39, 0.26))
R = np.diag((0.11, 0.11))

Q = np.diag((1, 1, 1, 1))
R = np.diag((1, 1))
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0, cost_to_go_cut=1000)

# initialize combined controller
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False,
)
controller.init()

# start simulation
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=[0, 0, 0.1, -0.1],
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=True,
)

# plot timeseries
plot_timeseries(
    T,
    X,
    U,
    X_meas=sim.meas_x_values,
    pos_y_lines=[np.pi],
    tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
    save_to="./timeseries_lqr.png",
)
