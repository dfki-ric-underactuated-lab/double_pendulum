import sys
import os
import shutil
import json
from tqdm import tqdm
import numpy as np
import jax

from double_pendulum.controller.DQN.exploration import EpsilonGreedySchedule
from double_pendulum.controller.DQN.replay_buffer import ReplayBuffer
from double_pendulum.controller.DQN.networks import DQN
from double_pendulum.controller.DQN.environment import get_environment
from double_pendulum.controller.DQN.simulate import simulate


def train(argvs) -> None:
    experiment_path = f"experiments/{argvs[0]}/"
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    if not os.path.exists(experiment_path + "parameters.json"):
        shutil.copy("parameters.json", experiment_path + "parameters.json")
    p = json.load(open(experiment_path + "parameters.json"))  # p for parameters

    sample_key, q_key, exploration_key = jax.random.split(jax.random.PRNGKey(p["seed"]), 3)
    n_training_steps = 0
    losses = np.zeros((p["n_epochs"], p["n_training_steps_per_epoch"])) * np.nan
    js = np.zeros(p["n_epochs"]) * np.nan
    max_j = -float("inf")
    argmax_j = None

    replay_buffer = ReplayBuffer(
        p["replay_buffer_size"],
        p["batch_size"],
        (4,),
        np.float32,
        lambda x: x,
    )

    env = get_environment(p["n_actions"])
    env.collect_random_samples(sample_key, replay_buffer, p["n_initial_samples"], p["horizon"])

    q = DQN(
        (4,),
        p["n_actions"],
        p["gamma"],
        p["layers"],
        q_key,
        p["learning_rate"],
        p["n_training_steps_per_online_update"],
        p["n_training_steps_per_target_update"],
    )

    epsilon_schedule = EpsilonGreedySchedule(
        p["starting_eps"],
        p["ending_eps"],
        p["duration_eps"],
        exploration_key,
        n_training_steps,
    )

    for idx_epoch in tqdm(range(p["n_epochs"])):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:
            sample_key, key = jax.random.split(sample_key)
            reward, has_reset = env.collect_one_sample(q, q.params, p["horizon"], replay_buffer, epsilon_schedule)

            sum_reward += reward
            n_episodes += int(has_reset)

            losses[
                idx_epoch, np.minimum(idx_training_step, p["n_training_steps_per_epoch"] - 1)
            ] = q.update_online_params(n_training_steps, replay_buffer, key)
            q.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        js[idx_epoch] = sum_reward / n_episodes
        np.save(
            f"{experiment_path}J_{p['seed']}.npy",
            js,
        )
        np.save(
            f"{experiment_path}L_{p['seed']}.npy",
            losses,
        )
        if js[idx_epoch] > max_j:
            if argmax_j is not None:
                os.remove(f"{experiment_path}Q_{p['seed']}_{argmax_j}_best_online_params")

            argmax_j = idx_epoch
            max_j = js[idx_epoch]
            q.save(f"{experiment_path}Q_{p['seed']}_{argmax_j}_best")

    simulate(experiment_path, env.actions)


if __name__ == "__main__":
    train(sys.argv[1:])
