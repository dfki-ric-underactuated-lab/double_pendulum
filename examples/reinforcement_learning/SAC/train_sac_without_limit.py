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

# setting log path for training
log_dir = "./log_data_designC.1/SAC_training"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# robot setup definition
# robot = "acrobot"
robot = "pendubot"

# model and reward parameter setting
max_velocity = 20
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
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
    design = "design_C.1"
    model = "model_1.0"
    load_path = "../../../data/controller_parameters/design_C.1/model_1.1/acrobot/lqr/"
    warm_start_path = ""
    # define para for quadratic reward
    Q = np.zeros((4, 4))
    Q[0, 0] = 100.0
    Q[1, 1] = 100.0
    Q[2, 2] = 1.0
    Q[3, 3] = 1.0
    R = np.array([[0.001]])
    r_line = 1e4
    r_vel = 1e3
    r_lqr = 1e5

# define model parameter path
model_par_path = (
        "../../../data/system_identification/identified_parameters/"
        + design
        + "/"
        + model
        + "/model_parameters.yml"
)

# load model parameters
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# load dynamics and simulator
dt = 0.01
integrator = "runge_kutta"
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

#===========================================================================================
# define learning parameters
state_representation = 2
obs_space=gym.spaces.Box(
            np.array([-1e4, -1e4, -1e4, -1e4]), np.array([1e4, 1e4, 1e4, 1e4])
        )
act_space=gym.spaces.Box(np.array([-1.0]), np.array([1.0]))
max_steps = 1000
termination = False

# define tuning parameters for training
n_envs = 100 # we found n_envs > 50 has very little improvement in training speed.
training_steps = 3e7 # default = 1e6
verbose = 1
reward_threshold = 1e10
eval_freq=5000
n_eval_episodes=5
learning_rate=0.01

#=========================================================================================
# initialize dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    scaling = False
)

# import LQR parameters for stabilization
rho = np.loadtxt(os.path.join(load_path, "rho"))
vol = np.loadtxt(os.path.join(load_path, "vol"))
S = np.loadtxt(os.path.join(load_path, "Smatrix"))

#===========================================================================================
## for testing
# warm_start = False
# print(robot)
# print(design)
# print(model)
# print("LQR load path=", load_path)
# print("model parameter path=", model_par_path)
# print("warm start status=", warm_start)
# if warm_start:
#     print("warm start path=",warm_start_path)

#============================================================================================
# defne roa checking function
def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    # calculate cost-to-go
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < rho, rad

# define reward_function without scaling
def reward_func(observation, action):
    """
    This function is the step reward function for reinforcement learning.

    Parameters
    ----------
    observation: np.array [pos1, pos2, vel1, vel2]
    action: np.array [torque1, torque2]

    Returns
    -------
    reward: scalar

    """
    # for testing
    # print("observation input into reward function=",observation)
    # print("action input into reward function=",action)

    ## define some hyperparameters
    control_line = 0.4 # stage 2 threshold

    # v_thresh = 8.0 # stage2 speed limit
    v_thresh = 10.0

    flag = False # reaching stage 2 flag

    vflag = False # stage 2 high speed flag

    bonus = False # reaching stage 3 flag

    ## translate the input value into values in physical world
    u = 5.0 * action # unscale action

    goal = [np.pi, 0., 0., 0.] # goal state

    y = wrap_angles_diff(observation) # make the pos1 and pos2 in [0,2pi] range

    ## check current state, and determine which flags are True
    # criterion 1: check if the end effector is above a certain line == in stage 2 or not
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

    # criteria 2: check if the state is in roa of the LQR controller == in stage 3 or not
    bonus, rad = check_if_state_in_roa(S, rho, y)

    # criteria 3: check if the velocity is larger than a preferred velocity in stage 2
    if flag and (np.abs(y[2]) > v_thresh or np.abs(y[3]) > v_thresh):
        vflag = True

    ## reward calculation
    # stage 1: quadratic reward
    r = np.einsum("i, ij, j", observation - goal, Q, observation - goal) + np.einsum("i, ij, j", u, R, u)
    reward = -1.0 * r

    # stage 2: end effector over a certain line
    if flag:
        print("stage1 reward=", reward)
        reward += r_line
        print("stage2 reward=", reward)
        # penalize on high velocity
        if vflag:
            print("oops")
            reward -= r_vel
        # stage 3: state in roa of the LQR controller
        if bonus:
            # roa method
            reward += r_lqr
            print("!!!bonus = True")

    return reward

def terminated_func(observation):
    # this is when termination takes place
    # check if the training episode should be terminated
    if np.abs(observation[0]) > np.pi * 4 or np.abs(observation[1]) > np.pi * 4:
        print("terminated!!!")
        return True
    else:
        return False

# def terminated_func(observation):
#     # this is when termination doesn't take place
#     return False

## reset function(both noisy and zero)
# this is when observation is unscaled
def noisy_reset_func():
    rand = np.random.rand(4) * 0.1
    rand[:] = rand[:] - 0.05
    observation = [0.0, 0.0, 0.0, 0.0] + rand
    return observation
def zero_reset_func():
    observation = [0.0, 0.0, 0.0, 0.0]
    return observation
#================================================
# # this when observation is scaled
# def noisy_reset_func():
#     rand = np.random.rand(4) * 0.01
#     rand[2:] = rand[2:] - 0.05
#     observation = [-1.0, -1.0, 0.0, 0.0] + rand
#     return observation
#
# def zero_reset_func():
#     observation = [-1.0, -1.0, 0.0, 0.0]
#     return observation

#========================================================================
## initialize different environments
# single training environment
env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
    scaling = False
)

# vectorized training environment
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
        "scaling":False
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
    scaling = False
)

#=============================================================
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

#================================================================
## final training process
# train
agent = SAC(
    MlpPolicy,
    envs,
    verbose=verbose,
    tensorboard_log=os.path.join(log_dir, "tb_logs"),
    learning_rate=learning_rate,
)

# warm start training
# warm_start = True
warm_start = False
if warm_start:
    agent.set_parameters(load_path_or_dict=warm_start_path)

agent.learn(total_timesteps=training_steps, callback=eval_callback)
