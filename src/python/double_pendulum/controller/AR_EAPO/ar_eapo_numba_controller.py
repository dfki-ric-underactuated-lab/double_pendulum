import time
from pathlib import Path

import numpy as np
import torch as th
from numba import jit, njit, float32

from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.AR_EAPO.ar_eapo import AR_EAPO


def to_numba(model: AR_EAPO):
    p = model.policy
    layers = [*p.mlp_extractor.policy_net.children(), p.action_net]

    ws, bs = [], []
    ss = []
    for l in layers:
        if isinstance(l, th.nn.Linear):
            ws.append(np.ascontiguousarray(l.weight.data.numpy().T))
            bs.append(np.ascontiguousarray(l.bias.data.numpy()))
            ss.append(ws[-1].shape)

    flat_weights = np.concatenate([w.ravel() for w in ws])
    flat_biases = np.concatenate(bs)

    return flat_weights, flat_biases, np.array(ss, dtype=np.int32)


@njit
def relu(x):
    return np.maximum(0, x)


def create_model(
    mu, sigma, c, weights, biases, shapes, tl, max_velocity, clip_velocity
):
    if not clip_velocity:

        @njit
        def predictor(x):
            x = x.copy()
            x[..., :2] = (x[..., :2] % (2 * np.pi) - np.pi) / np.pi
            x[..., 2:] = np.clip(x[..., 2:], -max_velocity, max_velocity) / max_velocity
            x = np.clip((x - mu) / sigma, -c, c)

            x = x.astype(np.float32)

            w_i = 0
            b_i = 0
            for i, s in enumerate(shapes):
                w_f = w_i + s[0] * s[1]
                b_f = b_i + s[1]
                w = weights[w_i:w_f].reshape(s[0], s[1])
                b = biases[b_i:b_f]
                x = np.dot(x, w) + b
                if i < len(shapes) - 1:
                    x = relu(x)
                w_i = w_f
                b_i = b_f
            a = np.tanh(x)
            return a * tl

    else:

        @njit
        def predictor(x):
            x = x.copy()
            x[..., :2] = (x[..., :2] % (2 * np.pi) - np.pi) / np.pi
            x[..., 2:] /= max_velocity
            x = np.clip((x - mu) / sigma, -c, c)

            x = x.astype(np.float32)

            w_i = 0
            b_i = 0
            for i, s in enumerate(shapes):
                w_f = w_i + s[0] * s[1]
                b_f = b_i + s[1]
                w = weights[w_i:w_f].reshape(s[0], s[1])
                b = biases[b_i:b_f]
                x = np.dot(x, w) + b
                if i < len(shapes) - 1:
                    x = relu(x)
                w_i = w_f
                b_i = b_f
            a = np.tanh(x)
            return a * tl

    return predictor


class NumbaJITController(AbstractController):
    def __init__(
        self,
        model_path: str | Path,
        robot: str,
        max_torque=6.0,
        max_velocity=20.0,
        clip_velocity=False,
        torque_compensation=0.5,
        device="cpu",
    ):
        """
        Initialize the NumbaJITController with a pre-trained model and configuration.

        Parameters
        ----------
        model_path : str | Path
            Path to the saved AR_EAPO model file to load.
        robot : str
            Type of robot to control, either "pendubot" or "acrobot". Determines which joint
            receives the active torque control.
        max_torque : float, default=6.0
            Maximum torque that can be applied by the controller, in N·m.
        max_velocity : float, default=20.0
            Maximum velocity used for observation normalization, in rad/s.
        clip_velocity : bool, default=False
            Whether to clip velocity values in observations. If True, velocities are clipped
            to [-max_velocity, max_velocity]. If False, velocities are only scaled by max_velocity.
        torque_compensation : float, default=0.5
            Additional torque compensation value applied to the unactuated joint, in N·m.
        device : str, default="cpu"
            Device to load the policy model on before conversion to Numba format.

        Notes
        -----
        The controller converts a PyTorch model to a Numba-optimized predictor function
        for efficient inference. This process includes:
        1. Loading and extracting network parameters (weights and biases)
        2. Setting up observation normalization using statistics from training
        3. Creating appropriate torque limits based on the robot configuration
        4. Compiling an optimized prediction function with Numba

        The controller automatically performs a warm-up call to ensure the JIT
        compilation is completed before the first control command.
        """
        super().__init__()
        model: AR_EAPO = AR_EAPO.load(model_path)
        model.policy = model.policy.to(device)
        vn = model.get_vec_normalize_env()
        print(model.policy.net_arch)
        assert vn is not None
        mu = vn.obs_rms.mean
        sigma = np.sqrt(vn.obs_rms.var + vn.epsilon)
        c = vn.clip_obs
        ws, bs, ss = to_numba(model)
        torque_limits = np.array(
            [
                max_torque * (robot == "pendubot")
                + torque_compensation * (robot == "acrobot"),
                max_torque * (robot == "acrobot")
                + torque_compensation * (robot == "pendubot"),
            ],
            np.float64,
        )
        self.predictor = create_model(
            mu, sigma, c, ws, bs, ss, torque_limits, max_velocity, clip_velocity
        )

        self.need_warm_up = True
        self.durations = []

        self.warm_up()

    @staticmethod
    @jit(nopython=True)
    def _normalise(x: np.ndarray, mu, sigma, c, max_velocity):
        x[..., :2] = (x[..., :2] % (2 * np.pi) - np.pi) / np.pi
        x[..., 2:] = np.clip(x[..., 2:], -max_velocity, max_velocity) / max_velocity

        x = np.clip((x - mu) / sigma, -c, c)

        return x

    def get_control_output_(self, x, t=None):
        # start = time.time()

        # a = self.predictor(x)

        # end = time.time()
        # elapsed = end - start
        # self.durations.append(elapsed)
        # return a
        return self.predictor(x)

    def reset_(self):
        self.durations.clear()

    def summary(self):
        m = np.mean(self.durations)
        h = np.max(self.durations)
        l = np.min(self.durations)
        print(l, m, h)

    def warm_up(self):
        dummy_input = np.random.randn(4)
        for _ in range(10):
            _ = self.predictor(dummy_input)


