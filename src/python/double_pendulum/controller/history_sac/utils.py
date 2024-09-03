from double_pendulum.utils.wrap_angles import wrap_angles_diff
import os

import numpy as np
from double_pendulum.simulation.gym_env import double_pendulum_dynamics_func
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator


class custom_dynamics_func_4PI(double_pendulum_dynamics_func):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_angle = 4 * np.pi

    def unscale_action(self, action):
        if isinstance(action, (float, list)):
            action = np.array(action).reshape(-1, 1)
        elif isinstance(action, np.ndarray) and action.shape == (2,):
            return action

        a = np.zeros((2,) if action.ndim == 1 else (action.shape[0], 2))

        if self.robot == "pendubot":
            a[..., 0] = self.torque_limit[0] * action[..., 0]
        elif self.robot == "acrobot":
            a[..., 1] = self.torque_limit[1] * action[..., -1]
        else:
            a = np.multiply(self.torque_limit, action[..., :2])

        return a.squeeze()

    def unscale_state(self, observation):
        observation = np.asarray(observation)
        scale = np.array([self.max_angle, self.max_angle, self.max_velocity, self.max_velocity])
        return observation * scale

    def normalize_state(self, state):
        state = np.asarray(state)
        angles = ((state[..., :2] + self.max_angle) % (2 * self.max_angle) - self.max_angle) / self.max_angle
        velocities = np.clip(state[..., 2:], -self.max_velocity, self.max_velocity) / self.max_velocity
        return np.concatenate([angles, velocities], axis=-1)


def load_param(torque_limit, simplify=True):
    design = "design_C.1"
    model = "model_1.1"
    torque_array = [torque_limit, torque_limit]

    current_dir = os.getcwd()
    double_pendulum_index = current_dir.find("double_pendulum")

    if double_pendulum_index != -1:
        base_path = current_dir[:double_pendulum_index + len("double_pendulum")]
        model_par_path = os.path.join(
            base_path,
            "data",
            "system_identification",
            "identified_parameters",
            design,
            model,
            "model_parameters.yml"
        )
    else:
        raise FileNotFoundError("The 'double_pendulum' folder was not found in the current working directory path.")

    mpar = model_parameters(filepath=model_par_path)
    mpar.set_torque_limit(torque_limit=torque_array)
    if simplify:
        mpar.set_motor_inertia(0.0)
        mpar.set_damping([0.0, 0.0])
        mpar.set_cfric([0.0, 0.0])

    return mpar


def general_dynamics(robot, dt, max_torque):
    max_vel = 30.0

    mpar = load_param(max_torque)
    plant = SymbolicDoublePendulum(model_pars=mpar)
    simulator = Simulator(plant=plant)

    dynamics_function = custom_dynamics_func_4PI(
        simulator=simulator,
        robot=robot,
        dt=dt,
        integrator="runge_kutta",
        max_velocity=max_vel,
        torque_limit=[max_torque, max_torque],
        scaling=True
    )
    return dynamics_function


def default_decider(obs, progress):
    return 1


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def column_softmax(x):
    col_sums = x.sum(axis=0)
    needs_softmax = ~np.isclose(col_sums, 1.0)
    x[:, needs_softmax] = softmax(x[:, needs_softmax])
    return x


def softmax_and_select(arr):
    softmax_probs = column_softmax(arr)
    result = np.zeros_like(arr)
    selected_rows = [np.argmax(np.random.multinomial(1, softmax_probs[:, i])) for i in range(arr.shape[1])]
    result[selected_rows, np.arange(arr.shape[1])] = 1
    return result


def find_observation_index(observation, history):
    """
        Finds the index of a specific observation in an array of observations.

        This function searches through the array of measurements stored in `history['X_meas']`
        and returns the index where the given `observation` is found. It compares each entry in
        reverse order to find the most recent match.
    """
    X_meas = history['X_meas']
    for idx, array in reversed(list(enumerate(X_meas))):
        if np.array_equal(observation, array.astype(np.float32)) or np.array_equal(observation, array):
            return idx
    return -1


def find_index_and_dict(observation, env):
    """
        Finds the index of the given observation in the environment's observation dictionaries.

        This function first attempts to find the observation in the current observation dictionary
        (`env.history`). If the observation is not found, it falls back to searching in an
        older observation dictionary (`env.history_old`).
    """
    history = env.history
    index = find_observation_index(observation, history)
    if index < 0:
        history = env.history_old
        index = find_observation_index(observation, history)

    return index, history


def get_unscaled_action(history, t_minus=0, key='U_real'):
    unscaled_action = history['dynamics_func'].unscale_action(history[key][t_minus-1])
    max_value_index = np.argmax(np.abs(unscaled_action))
    max_action_value = unscaled_action[max_value_index]
    return max_action_value


def get_state_values(history, key='X_meas', offset=0):
    l = history['mpar'].l
    unscaled_observation = history['dynamics_func'].unscale_state(history[key][offset-1])

    y = wrap_angles_diff(unscaled_observation)  #now both angles from -pi to pi

    s1 = np.sin(y[0])
    s2 = np.sin(y[0] + y[1])
    c1 = np.cos(y[0])
    c2 = np.cos(y[0] + y[1])

    #cartesians of elbow x1 and end effector x2
    x1 = np.array([s1, c1]) * l[0]
    x2 = x1 + np.array([s2, c2]) * l[1]

    #cartesian velocities of the joints
    v1 = np.array([c1, -s1]) * y[2] * l[0]
    v2 = v1 + np.array([c2, -s2]) * (y[2] + y[3]) * l[1]

    state_values = {
        "x2": x2,
        "v2": v2,
        "c1": c1,
        "c2": c2,
    }

    return state_values