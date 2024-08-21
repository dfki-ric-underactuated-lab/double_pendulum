import gymnasium as gym
import numpy as np

class CustomCustomEnv(gym.Env):
    def __init__(
        self,
        dynamics_func,
        reset_func,
        obs_space=gym.spaces.Box(
            np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
        ),
        act_space=gym.spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        max_episode_steps=1000,
        scaling=True,
        terminates=True,
    ):
        self.dynamics_func = dynamics_func
        self.reset_func = reset_func
        self.observation_space = obs_space
        self.action_space = act_space
        self.max_episode_steps = max_episode_steps

        self.previous_action = 0
        self.terminates = terminates

        self.observation = self.reset_func()
        self.step_counter = 0
        self.stabilisation_mode = False
        self.y = [0,0] 
        self.update_y()
        self.scaling = scaling

        l1 = self.dynamics_func.simulator.plant.l[0]
        l2 = self.dynamics_func.simulator.plant.l[1]
        self.max_height = l1 + l2

        if self.dynamics_func.robot == "acrobot":
            self.control_line = 0.75 * self.max_height
        elif self.dynamics_func.robot == "pendubot":
            self.control_line = 0.7 * self.max_height

        self.old_obs = None

    def step(self, action):
        self.old_obs = np.copy(self.observation)
        self.observation = self.dynamics_func(
            self.observation, action, scaling=self.scaling
        )

        self.update_y()
        self.stabilisation_mode = self.y[1] >= self.control_line
        terminated = self.terminated_func()
        reward = self.reward_func(terminated, action)
        info = {}
        truncated = False
        self.step_counter += 1
        if self.step_counter >= self.max_episode_steps:
            truncated = True
        self.previous_action = action[0]
        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observation = self.reset_func()
        self.step_counter = 0
        info = {}
        self.previous_action = 0
        self.stabilisation_mode = False
        self.old_obs = np.copy(self.observation)
        return self.observation, info

    def render(self, mode="human"):
        pass

    def reward_func(self, terminated, action):
        raise NotImplementedError("You have to define the reward function")

    def terminated_func(self):
        if self.terminates:
            # Checks if we're in stabilisation mode and the ee has fallen below the control line
            if self.stabilisation_mode and self.y[1] < self.control_line:
                return True
        return False

    # Update the y coordinate of the first joint and the end effector
    def update_y(self):
        theta1, theta2, _, _ = self.dynamics_func.unscale_state(self.observation)

        link_end_points = self.dynamics_func.simulator.plant.forward_kinematics(
            [theta1, theta2]
        )
        self.y[0] = link_end_points[0][1]
        self.y[1] = link_end_points[1][1]

    def gravitational_reward(self):
        x = self.dynamics_func.unscale_state(self.observation)
        V = self.dynamics_func.simulator.plant.potential_energy(x)
        return V

    def V(self):
        return self.gravitational_reward()

    def kinetic_reward(self):
        x = self.dynamics_func.unscale_state(self.observation)
        T = self.dynamics_func.simulator.plant.kinetic_energy(x)
        return T

    def T(self):
        return self.kinetic_reward()
