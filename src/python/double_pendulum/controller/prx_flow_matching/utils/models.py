import torch

import os

from .json_args import JSONArgs
from ..models.flow_matching import *
from ..models.temporal import *

def get_model_classes(model_args):
    model_class = model_args.model.split(".")[-1]
    fm_class = model_args.flow_matching.split(".")[-1]

    if model_class == "TemporalUnet":
        model_class = TemporalUnet
    elif model_class == "TemporalTransformer":
        model_class = TemporalTransformer
    else:
        raise ValueError(f"Model class {model_class} not found")

    if fm_class == "FlowMatching":
        fm_class = FlowMatching
    else:
        raise ValueError(f"Flow matching class {fm_class} not found")

    return model_class, fm_class
        

def load_model(model_path):
    model_args = JSONArgs(os.path.join(model_path, "args.json"))
    model_path = os.path.join(model_path, "best.pt")

    fm_model_state_dict = torch.load(model_path, weights_only=False)

    model_class, fm_class = get_model_classes(model_args)

    model = model_class(
        horizon_length=model_args.horizon_length + model_args.history_length,
        transition_dim=model_args.observation_dim,
        cond_dim=model_args.observation_dim,
        **model_args.model_kwargs
    ).to('cuda')

    fm_model = fm_class(
        model=model,
        observation_dim=model_args.observation_dim,
        history_length=model_args.history_length,
        horizon_length=model_args.horizon_length,
        clip_denoised=model_args.clip_denoised,
        loss_type=model_args.loss_type,
        loss_weights=model_args.loss_weights,
        loss_discount=model_args.loss_discount,
        action_indices=model_args.action_indices,
    ).to('cuda')

    if "model" in fm_model_state_dict:
        fm_model.load_state_dict(fm_model_state_dict["model"])
    else:
        fm_model.load_state_dict(fm_model_state_dict, strict=False)
    

    return fm_model, model_args
