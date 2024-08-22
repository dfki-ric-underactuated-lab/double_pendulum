import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.simulation.gym_env import (
    CustomEnv,
    double_pendulum_dynamics_func,
)
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.wrap_angles import wrap_angles_diff

# setting log path for the training
log_dir = "./log_data/SAC_training"
# log_dir = "./log_data_designC.1/SAC_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# define robot variation
robot = "acrobot"
# robot = "pendubot"

# model and reward parameter
max_velocity = 20
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    # design A.0
    # design = "design_A.0"
    # model = "model_2.0"
    # load_path = "../../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr/"
    # warm_start_path = ""
    # Q = np.zeros((4, 4))
    # Q[0, 0] = 8.0
    # Q[1, 1] = 5.0
    # Q[2, 2] = 0.1
    # Q[3, 3] = 0.1
    # R = np.array([[0.0001]])
    # r_line = 500
    # r_vel = 0
    # r_lqr = 1e4

    # design C.1
    design = "design_C.1"
    model = "model_1.0"
    load_path = "../../../data/controller_parameters/design_C.1/model_1.1/pendubot/lqr/"
    warm_start_path = ""
    # define para for quadratic reward
    Q = np.zeros((4, 4))
    Q[0, 0] = 100.0
    Q[1, 1] = 100.0
    Q[2, 2] = 1.0
    Q[3, 3] = 1.0
    R = np.array([[0.01]])
    r_line = 1e3
    r_vel = 0
    r_lqr = 1e5


elif robot == "acrobot":
    torque_limit = [0.0, 5.0]

    # design C.0
    # design = "design_C.0"
    # model = "model_3.0"
    # load_path = "../../../data/controller_parameters/design_C.0/acrobot/lqr/roa"
    # warm_start_path = ""
    # Q = np.zeros((4, 4))
    # Q[0, 0] = 10.0
    # Q[1, 1] = 10.0
    # Q[2, 2] = 0.2
    # Q[3, 3] = 0.2
    # R = np.array([[0.0001]])
    # r_line = 500
    # r_vel = 1e4
    # r_lqr = 1e4

    # design C.1
    design = "design_C.1"
    model = "model_1.0"
    load_path = "../../../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr/"
    warm_start_path = ""
    # define para for quadratic reward
    Q = np.zeros((4, 4))
    Q[0, 0] = 100.0
    Q[1, 1] = 105.0
    Q[2, 2] = 1.0
    Q[3, 3] = 1.0
    R = np.array([[0.01]])
    r_line = 1e3
    r_vel = 1e4
    r_lqr = 1e5

model_par_path = (
        "../../../data/system_identification/identified_parameters/"
        + design
        + "/"
        + model
        + "/model_parameters.yml"
)

mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)
dt = 0.01
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 2
obs_space = gym.spaces.Box(
    np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
)
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
max_steps = 1000
termination = False
############################################################################

#tuning parameter
n_envs = 100 # we found n_envs > 50 has very little improvement in training speed.
training_steps = 3e7 # default = 1e6
verbose = 1
# reward_threshold = -0.01
reward_threshold = 1e10
eval_freq=2500
n_eval_episodes=10
learning_rate=0.01
##############################################################################
# initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
)

# import lqr parameters
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))

def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad

def reward_func(observation, action):
    # define reward para according to robot type
    control_line = 0.4
    v_thresh = 8.0
    # v_thresh = 10.0
    vflag = False
    flag = False
    bonus = False

    # state
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * max_velocity,
            observation[3] * max_velocity,
        ]
    )

    u = 5.0 * action

    goal = [np.pi, 0., 0., 0.]

    y = wrap_angles_diff(s)

    # criterion 1: control line
    p1 = y[0]
    p2 = y[1]
    ee1_pos_x = 0.2 * np.sin(p1)
    ee1_pos_y = -0.2 * np.cos(p1)

    ee2_pos_x = ee1_pos_x + 0.3 * np.sin(p1 + p2)
    ee2_pos_y = ee1_pos_y - 0.3 * np.cos(p1 + p2)
    # print(ee2_pos_y)
    if ee2_pos_y >= control_line:
        flag = True
        print("flag=", flag)
        # print(ee2_pos_y)
    else:
        flag = False

    # criteria 2: roa check
    bonus, rad = check_if_state_in_roa(S, rho, y)

    # criteria 3: velocity check
    if flag and (np.abs(y[2]) > v_thresh or np.abs(y[3]) > v_thresh):
        vflag = True


    # reward calculation
    ## stage1: quadratic reward
    r = np.einsum("i, ij, j", s - goal, Q, s - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r

    ## stage2: control line reward
    if flag:
        print("stage1 reward=",reward)
        reward += r_line
        print("stage2 reward=", reward)
        ## stage 3: roa reward
        if bonus:
            # roa method
            reward += r_lqr
            print("!!!bonus = True")
        ## penalize on high velocity
        if vflag:
            print("oops")
            reward -= r_vel
    else:
        reward = reward

    return reward

def terminated_func(observation):
    s = np.array(
        [
            observation[0] * np.pi + np.pi,  # [0, 2pi]
            (observation[1] * np.pi + np.pi + np.pi) % (2 * np.pi) - np.pi,  # [-pi, pi]
            observation[2] * max_velocity,
            observation[3] * max_velocity,
        ]
    )
    # y = wrap_angles_top(s)
    y = wrap_angles_diff(s)
    bonus, rad = check_if_state_in_roa(S, rho, y)
    if termination:
        if bonus:
            print("terminated")
            return bonus
    else:
        return False

def noisy_reset_func():
    rand = np.random.rand(4) * 0.01
    rand[2:] = rand[2:] - 0.05
    observation = [-1.0, -1.0, 0.0, 0.0] + rand
    return observation

def zero_reset_func():
    observation = [-1.0, -1.0, 0.0, 0.0]
    return observation

# initialize vectorized environment
env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training env
envs = make_vec_env(
    env_id=CustomEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reward_func": reward_func,
        "terminated_func": terminated_func,
        "reset_func": noisy_reset_func,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
    },
)

# evaluation env
eval_env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

# training callbacks
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=os.path.join(log_dir,
                                      "best_model"),
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

# train
agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
    learning_rate=learning_rate,
)

# warm_start = True
warm_start = False
if warm_start:
    agent.set_parameters(load_path_or_dict=warm_start_path)

agent.learn(total_timesteps=training_steps, callback=eval_callback)


