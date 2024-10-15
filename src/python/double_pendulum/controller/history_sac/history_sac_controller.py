from double_pendulum.controller.abstract_controller import AbstractController
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

from double_pendulum.controller.history_sac.utils import general_dynamics
from double_pendulum.controller.history_sac.history_sac import HistorySAC, HistoryEnv

def resample_and_denoise(dt, times, values, force_edges=False, max_length=None):
    """
    Resample the time series with a coarser resolution and reduce noise by averaging surrounding data points.

    Parameters:
    times (list of float): Original time points.
    values (list of float or list of np.ndarray): Corresponding values for the time points.
    dt (float): The coarser time step for resampling.
    force_edges (bool): Whether to set the first and last resampled values to the original values.
    max_length (int, optional): Maximum length of the output lists. The function terminates early if the limit is reached.

    Returns:
    resampled_times (list of float): Resampled time points.
    resampled_values (list): Values for the resampled time points.
                             A list of floats if input values are a list of floats,
                             or a list of np.ndarrays if input values are a list of numpy arrays.
    """

    if len(times) == 1:
        return times, values

    times = np.array(times)
    values = np.array(values)

    # Initialize lists for resampled data
    resampled_times = []
    resampled_values = []

    is_array_input = isinstance(values[0], np.ndarray)

    # Start from the last time point and move backwards
    current_time = times[-1]

    while current_time >= times[0]:
        indices = np.where((times >= current_time - dt) & (times <= current_time))[0]

        if len(indices) > 0:
            recent_value = values[indices[-1]][:2]
            avg_last_two_channels = np.mean(values[indices][:, 2:], axis=0)
            avg_value = np.concatenate([recent_value, avg_last_two_channels])
            resampled_times.append(current_time)
            resampled_values.append(avg_value)

        # Move backwards by dt
        current_time -= dt

        # Terminate early if the max_length is reached
        if max_length is not None and len(resampled_times) >= max_length:
            break

    # Reverse the lists to have them in increasing time order
    resampled_times = resampled_times[::-1]
    resampled_values = resampled_values[::-1]

    # Handle the case where resampled_times has a length of 1
    if len(resampled_times) == 1 and force_edges:
        resampled_times[0] = times[-1]
        resampled_values[0] = values[-1]
        return resampled_times, resampled_values

    # Force the first and last values if required
    if force_edges and len(resampled_times) > 1:
        # Force the last value to match the original last value
        resampled_times[-1] = times[-1]
        resampled_values[-1] = values[-1]

        # Force the first value to match the original first value if possible
        if resampled_times[0] == times[0]:
            resampled_values[0] = values[0]

    return resampled_times, resampled_values

class HistorySACController(AbstractController):
    def __init__(self, env_type, model_path, lowpass=0.0):
        super().__init__()

        self.lowpass = lowpass
        dt = 0.02
        max_torque = 6
        dynamics_func = general_dynamics(env_type, dt, max_torque)

        envs = make_vec_env(
            env_id=HistoryEnv,
            n_envs=1,
            env_kwargs={
                "dynamic_function": dynamics_func,
                "reward_function": None,
                "termination_function": None,
                "reset_function": lambda: [0, 0, 0, 0],
                "torque_limit": max_torque
            },
            vec_env_cls=DummyVecEnv
        )

        self.model = HistorySAC.load(
            model_path,
            env=envs,
            print_system_info=True,
            env_type=env_type
        )

        self.dynamics_func = dynamics_func
        self.simulator = self.dynamics_func.simulator
        self.dt = self.dynamics_func.dt
        self.scaling = dynamics_func.scaling
        self.integrator = dynamics_func.integrator
        self.history = None
        self.last_action = None
        self.last_u = None
        self.reset()

    def reset(self):
        super().reset()
        self.history = {'T': [], 'X': [], 'U': []}
        self.last_action = 0.0
        self.last_u = None
        self.model.env.envs[0].env.reset()

    def get_control_output_(self, x, t=None):

        self.history['T'].append(np.round(t, decimals=5))
        env = self.model.env.envs[0].env
        obs = self.dynamics_func.normalize_state(x)

        if np.rint(t * 10000) % np.rint(self.dt * 10000) == 0 and t > 0.0:
            env.history['T'].append(np.round(t, decimals=5))

        self.history['X'].append(obs)
        self.history['U'].append(self.last_action)

        _, env.history['X_meas'] = resample_and_denoise(self.dt, self.history['T'], self.history['X'], force_edges=False, max_length=12)

        action, _ = self.model.predict(observation=env.history['X_meas'][-1].reshape(1, -1), deterministic=True)
        lowpass = self.lowpass
        if self.last_action == 0.0:
            lowpass = 0.0
        new_action = lowpass * self.last_action + (1 - lowpass) * action.item()
        self.last_action = new_action
        return self.dynamics_func.unscale_action(new_action)


class IdentificationController(AbstractController):
    def __init__(self, joint, f, a):
        super().__init__()

        self.joint = joint
        self.f = f
        self.a = a

    def get_control_output_(self, x, t=None):
        u = np.sin(2 * np.pi * self.f * t) * self.a
        action = [0, 0]
        action[self.joint] = u
        return np.array(action)
