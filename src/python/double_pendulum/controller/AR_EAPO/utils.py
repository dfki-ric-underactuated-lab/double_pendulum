from typing import Sequence

import torch as th
import torch.nn as nn


def parse_activation_fn(name: str):
    try:
        activation: type[nn.Module] = getattr(nn.modules.activation, name)
    except AttributeError as e:
        print("utils.build_mlp: ", e)
        raise RuntimeError from e
    return activation


def build_mlp(
    input_dim: int,
    output_dim: int | None,
    net_arch: Sequence[int],
    activation_fn: str | type[nn.Module],
):
    if isinstance(activation_fn, str):
        activation = parse_activation_fn(activation_fn)
    else:
        activation = activation_fn

    net: list[nn.Module] = []
    last_dim = input_dim
    for dim in net_arch:
        net.append(nn.Linear(last_dim, dim))
        net.append(activation())
        last_dim = dim
    if output_dim:
        net.append(nn.Linear(last_dim, output_dim))

    return nn.Sequential(*net)
