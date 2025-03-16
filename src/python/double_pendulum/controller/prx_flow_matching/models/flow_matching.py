from collections import namedtuple

from torch import nn
import torch

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

from .helpers import (
    apply_conditioning
)
from .helpers.losses import Losses


Sample = namedtuple("Sample", "horizon values chains")


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    """
    Get the model_mean and the fixed variance from the model

    then sample noise from a normal distribution
    """
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class FlowMatching(nn.Module):
    def __init__(
        self,
        model,
        observation_dim,
        horizon_length,
        history_length,
        clip_denoised=False,
        loss_type="l1",
        loss_weights=None,
        loss_discount=1.0,
        action_indices=None,
    ):
        super().__init__()

        self.model = model

        self.observation_dim = observation_dim
        self.clip_denoised = clip_denoised
        self.horizon_length = horizon_length
        self.history_length = history_length
        self.action_indices = action_indices
        loss_weights = self.get_loss_weights(loss_discount, loss_weights)

        self.loss_weights = loss_weights

        self.loss_fn = Losses[loss_type](loss_weights, history_length, action_indices)

        # Setup flow matching components
        self.scheduler = CondOTScheduler()
        self.path = AffineProbPath(self.scheduler)
        self.solver = ODESolver(self.vector_field)

    def vector_field(self, x=None, t=None):
        if x is None or t is None:
            raise ValueError("x and t must be provided")
        
        if t.ndim == 0:
            t = t.unsqueeze(0)

        if x.ndim == 2:
            x = x.unsqueeze(0)

        return self.model(x, None, t)

    def get_loss_weights(self, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''

        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon_length + self.history_length, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        return loss_weights
    
    # ------------------------------------------ training ------------------------------------------#

    def loss(self, x_target, cond):
        """
        Choose a random timestep t and calculate the loss for the model
        """
        batch_size = len(x_target)
        t = torch.rand(batch_size, device=x_target.device)

        # Move loss weights to the same device as input tensors if needed
        if self.loss_weights.device != x_target.device:
            self.loss_weights = self.loss_weights.to(x_target.device)

        x_noisy = torch.randn_like(x_target)
        x_noisy = apply_conditioning(x_noisy, cond)

        path_sample = self.path.sample(t=t, x_0=x_noisy, x_1=x_target)

        loss, info = self.loss_fn(self.vector_field(x=path_sample.x_t, t=path_sample.t), path_sample.dx_t, loss_weights=self.loss_weights)

        return loss, info

    
    # ------------------------------------------ inference ------------------------------------------#

    @torch.no_grad()
    def conditional_sample(self, cond, step_size=0.05, n_timesteps=100, integration_method="midpoint", return_intermediates=False, device='cuda'):
        """
        conditions : [ (time, state), ... ]
        """
        horizon = self.horizon_length + self.history_length
        shape = (horizon, self.observation_dim)

        T = torch.linspace(0, 1, n_timesteps)

        x_noisy = torch.randn(shape, device=device)
        x_noisy = apply_conditioning(x_noisy, cond)

        sol = self.solver.sample(x_noisy, step_size=step_size, time_grid=T, method=integration_method, return_intermediates=return_intermediates)

        horizon = sol[self.history_length:]

        if self.clip_denoised:
            horizon = torch.clamp(horizon, -1.0, 1.0)

        return Sample(horizon=horizon, values=None, chains=None)
    
    
    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
