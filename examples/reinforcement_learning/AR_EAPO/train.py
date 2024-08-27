from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from double_pendulum.controller.AR_EAPO.ar_eapo import AR_EAPO, ARConfig, EAPOConfig

from environment import make_env, ROBOTS


@dataclass
class EnvConfig:
    robot: ROBOTS
    model_params_path: str | Path
    dt: float
    max_torque: float
    max_velocity: float
    scaling: bool
    max_steps: int
    Q: list[float]
    R: float
    reward_scale: float
    random_truncation_probability: float
    n_envs: int
    norm_obs: bool
    seed: int | None = None


@dataclass
class PPOConfig:
    learning_rate: float = 5e-4
    n_steps: int = 128
    batch_size: int = 1024
    n_epochs: int = 6
    gamma: float = 1.0
    gae_lambda: float = 0.8
    clip_range: float = 0.05
    normalize_advantage: bool = True
    vf_coef: float = 0.25
    max_grad_norm: float = 10.0
    policy_kwargs: dict[str, Any] | None = None


@dataclass
class Config:
    env_config: EnvConfig
    ar_config: ARConfig = ARConfig()
    eapo_config: EAPOConfig = EAPOConfig()
    ppo_config: PPOConfig = PPOConfig()
    device: str = "cuda"
    verbose: int = 1
    tensorboard_log: str | None = None
    seed: int | None = None
    total_timesteps: int = int(3e7)
    model_save_dir: Path | str | None = None


def train(config: Config, _=None):
    env, dynamics_func = make_env(**asdict(config.env_config))

    model = AR_EAPO(
        env,
        config.ar_config,
        config.eapo_config,
        **asdict(config.ppo_config),
        device=config.device,
        seed=config.seed,
        tensorboard_log=config.tensorboard_log,
        verbose=config.verbose,
    )

    model.learn(config.total_timesteps, tb_log_name="AR_EAPO")

    if config.model_save_dir is not None:
        model_save_dir = Path(config.model_save_dir)
        model_save_dir = (
            model_save_dir
            / config.env_config.robot
            / f"{datetime.now().strftime('%y%m%d_%H%M%S')}"
        )
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_save_dir / "model.zip")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", "-c", default="config.yaml")
    args = parser.parse_args()

    with open(args.config_path) as f:
        yaml_dict = yaml.load(f, yaml.FullLoader)

    yaml_dict["env_config"] = EnvConfig(**yaml_dict["env_config"])
    yaml_dict["ar_config"] = ARConfig(**yaml_dict.get("ar_config", {}))
    yaml_dict["eapo_config"] = EAPOConfig(**yaml_dict.get("eapo_config", {}))
    yaml_dict["ppo_config"] = PPOConfig(**yaml_dict.get("ppo_config", {}))
    config = Config(**yaml_dict)

    train(config, "🍕")
