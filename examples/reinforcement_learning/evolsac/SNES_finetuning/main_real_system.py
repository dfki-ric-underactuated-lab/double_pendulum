import os
import sys

import pandas
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
import stable_baselines3
import torch
from magic import MagicCallback, BruteMagicCallback, load_controller
from simulator import CustomSimulator
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from wrappers import *

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.analysis.leaderboard import leaderboard_scores
import shutil

np.random.seed(0)
torch.manual_seed(0)
torch.random.manual_seed(0)
torch.backends.cudnn.deterministic = True
stable_baselines3.common.utils.set_random_seed(0)



def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                shutil.rmtree(directory_path)
    except OSError as e:
        print("Error occurred while deleting files:\n" + str(e))

assert (
    len(sys.argv) >= 2
), "Please provide: [max torque] [robustness] [window_size (0 = no window)] [include_time]"
max_torque = 3.0
WINDOW_SIZE = 0
INCLUDE_TIME = False
robot = str(sys.argv[1])


design = "design_C.1"
model = "model_1.1"
model_par_path = (
    "../../../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)

# model and reward parameter
max_velocity = 50
torque_limit = [max_torque, 0] if robot == "pendubot" else [0, max_torque]

mpar = model_parameters(filepath=model_par_path)
mpar.set_torque_limit(torque_limit)
dt = 0.01
t_final = 10
integrator = "runge_kutta"

plant = SymbolicDoublePendulum(model_pars=mpar)
eval_simulator = Simulator(plant=plant)

# learning environment parameters
state_representation = 2

obs_space = gym.spaces.Box(np.array([-1.0] * 4), np.array([1.0] * 4))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
max_steps = 10 / dt

###############################################################################
# initialize double pendulum dynamics
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot=robot,
    state_representation=state_representation,
    max_velocity=max_velocity,
    torque_limit=torque_limit,
)
###############################################################################

n_envs = 1
training_steps = 30_000_000_000_000
verbose = 1
eval_freq = 10_000
n_eval_episodes = 1
# a patto che i reward istantanei siano piccoli
# 0.01 -> 1500000 -> 7
# 0.003 -> 1500000 -> 46
# 0.001 -> 1500000 -> 38
# 0.0003 -> 1500000 -> 19
learning_rate = 0.001

model_path = "../../../../data/policies/design_C.1/model_1.0/pendubot/evolsac/model.zip"
REFERENCE_AGENT = SAC.load(model_path, device=torch.device("cpu"))

from evotorch import Problem
from evotorch.algorithms.distributed.gaussian import SNES
from magic import deepcopy_model, magic_score

WORKERS = 1
i = 0
j = 0
import uuid

def run_policy(state_dict, n_experiments):
    global j
    agent = deepcopy_model(REFERENCE_AGENT)
    with torch.no_grad():
        agent.policy.actor.latent_pi.load_state_dict(state_dict)

    # Load Controller
    controller, leaderboard_config = load_controller(
        dynamics_func, model, WINDOW_SIZE, INCLUDE_TIME, evaluating=False
    )

    # Run n experiments
    scores = []
    for n in range(n_experiments):
        save_dir = os.path.join("snes_log", design, model, robot, str(j)+"_test", str(n))
        experiment_done = False
        while not experiment_done:
            print(save_dir)

            # run_experiment(
            #     controller=controller,
            #     dt=dt,
            #     t_final=t_final,
            #     can_port="can0",
            #     motor_ids=[3, 1],
            #     motor_directions=[1.0, -1.0],
            #     tau_limit=torque_limit,
            #     save_dir=save_dir,
            #     record_video=True,
            #     safety_velocity_limit=30.0,
            #     # perturbation_array=perturbation_array,
            # )

            dirs = [x[0] for x in os.walk(save_dir)]
            print(dirs)
            if len(dirs) > 0 and os.path.isfile(dirs[1] + "/trajectory.csv"):
                experiment_done = True
            else:
                # delete trash
                delete_files_in_directory(save_dir)

        conf = leaderboard_config
        conf["csv_path"] = os.path.join(dirs[1], "trajectory.csv")
        data_paths = {}
        data_paths[leaderboard_config["name"]] = conf

        save_to = os.path.join(dirs[1], "leaderboard_entry.csv")

        leaderboard_scores(
            data_paths=data_paths,
            save_to=save_to,
            mpar=mpar,
            # weights={"swingup_time": 0.5, "max_tau": 0.1, "energy": 0.0, "integ_tau": 0.4, "tau_cost": 0.0, "tau_smoothness": 0.0},
            weights={
                "swingup_time": 1.0,
                "max_tau": 0.0,
                "energy": 1.0,
                "integ_tau": 0.0,
                "tau_cost": 1.0,
                "tau_smoothness": 1.0,
                "velocity_cost": 1.0,
            },
            normalize={
                "swingup_time": 20.0,
                "max_tau": 1.0,  # not used
                "energy": 60.0,
                "integ_tau": 1.0,  # not used
                "tau_cost": 20.0,
                "tau_smoothness": 0.1,
                "velocity_cost": 400,
            },
            link_base="",
            score_version="v2",
        )
        df = pandas.read_csv(save_to)
        scores.append(np.array(df["RealAI Score"])[0])
    # Get average score
    score = np.mean(scores)

    j += 1

    return score if not np.isnan(score) else 0.0


def test_policy(policy_params, n_experiments=5):
    global i
    IDX = i

    agent = deepcopy_model(REFERENCE_AGENT)
    with torch.no_grad():
        state_dict = agent.policy.actor.latent_pi.state_dict()
        keys = list(state_dict.keys())
        split_sizes = [torch.numel(state_dict[key]) for key in keys]
        params_split = torch.split(policy_params.clone().detach(), split_sizes)
        state_dict.update(
            {
                key: param.reshape(state_dict[key].shape)
                for key, param in zip(keys, params_split)
            }
        )
        agent.policy.actor.latent_pi.load_state_dict(state_dict)

    index = uuid.uuid4().hex

    # Load Controller
    controller, leaderboard_config = load_controller(
        dynamics_func, model, WINDOW_SIZE, INCLUDE_TIME, evaluating=False
    )

    # Run n experiments
    scores = []
    for n in range(n_experiments):
        save_dir = os.path.join("snes_log", design, model, robot, str(IDX), str(n))
        experiment_done = False
        while not experiment_done:
            print(save_dir)

            # run_experiment(
            #     controller=controller,
            #     dt=dt,
            #     t_final=t_final,
            #     can_port="can0",
            #     motor_ids=[3, 1],
            #     motor_directions=[1.0, -1.0],
            #     tau_limit=torque_limit,
            #     save_dir=save_dir,
            #     record_video=True,
            #     safety_velocity_limit=30.0,
            #     # perturbation_array=perturbation_array,
            # )

            dirs = [x[0] for x in os.walk(save_dir)]
            print(dirs)
            if len(dirs) > 0 and os.path.isfile(dirs[1] + "/trajectory.csv"):
                experiment_done = True
            else:
                # delete trash
                delete_files_in_directory(save_dir)

        conf = leaderboard_config
        conf["csv_path"] = os.path.join(dirs[1], "trajectory.csv")
        data_paths = {}
        data_paths[leaderboard_config["name"]] = conf

        save_to = os.path.join(dirs[1], "leaderboard_entry.csv")

        leaderboard_scores(
            data_paths=data_paths,
            save_to=save_to,
            mpar=mpar,
            # weights={"swingup_time": 0.5, "max_tau": 0.1, "energy": 0.0, "integ_tau": 0.4, "tau_cost": 0.0, "tau_smoothness": 0.0},
            weights={
                "swingup_time": 1.0,
                "max_tau": 0.0,
                "energy": 1.0,
                "integ_tau": 0.0,
                "tau_cost": 1.0,
                "tau_smoothness": 1.0,
                "velocity_cost": 1.0,
            },
            normalize={
                "swingup_time": 20.0,
                "max_tau": 1.0,  # not used
                "energy": 60.0,
                "integ_tau": 1.0,  # not used
                "tau_cost": 20.0,
                "tau_smoothness": 0.1,
                "velocity_cost": 400,
            },
            link_base="",
            score_version="v2",
        )
        df = pandas.read_csv(save_to)
        scores.append(np.array(df["RealAI Score"])[0])
    # Get average score
    score = np.mean(scores)
    print(score)

    i += 1

    return score if not np.isnan(score) else 0.0


# Set up the EvoTorch problem
problem = Problem(
    "max",
    lambda par: test_policy(par, n_experiments=1),
    solution_length=len(
        torch.cat(
            [
                p.data.view(-1)
                for p in REFERENCE_AGENT.policy.actor.latent_pi.parameters()
            ]
        )
    ),
    num_actors=WORKERS,
)

initial_solution = np.concatenate(
    [
        p.data.cpu().numpy().flatten()
        for p in REFERENCE_AGENT.policy.actor.latent_pi.parameters()
    ]
)

optimizer = SNES(problem, popsize=WORKERS, center_init=initial_solution, stdev_init=0.0075)
for generation in range(1000):
    optimizer.step()
    print(
        f"Generation {generation}: Best reward so far: {optimizer.status['best'].evals}"
    )

    best_params = optimizer.status["best"].values
    score = optimizer.status["best_eval"]

    # Update the policy's parameters
    with torch.no_grad():
        state_dict = REFERENCE_AGENT.policy.actor.latent_pi.state_dict()
        keys = list(state_dict.keys())
        split_sizes = [torch.numel(state_dict[key]) for key in keys]
        params_split = torch.split(best_params.clone().detach(), split_sizes)
        state_dict.update(
            {
                key: param.reshape(state_dict[key].shape)
                for key, param in zip(keys, params_split)
            }
        )
        REFERENCE_AGENT.policy.actor.latent_pi.load_state_dict(state_dict)

    real_score = run_policy(REFERENCE_AGENT.policy.actor.latent_pi.state_dict(), n_experiments=10)

    # Save the model with the best parameters
    import os

    if not os.path.exists(f"savings_{max_torque}"):
        os.makedirs(f"savings_{max_torque}", exist_ok=True)
    REFERENCE_AGENT.save(f"savings_{max_torque}/{generation}/best_model-{score}-{real_score}.zip")
    torch.save(
        REFERENCE_AGENT.policy.actor.latent_pi.state_dict(),
        f"savings_{max_torque}/{generation}/optimized_policy-{score}-{real_score}.pth",
    )

