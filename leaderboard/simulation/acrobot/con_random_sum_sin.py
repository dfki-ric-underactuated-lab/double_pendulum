from double_pendulum.controller.random_exploration.random_exploration_controller import Controller_Random_exploration

from sim_parameters import (
    mpar,
    dt,
    t_final,
    t0,
    x0,
    goal,
    integrator,
    design,
    model,
    robot,
)

name = "random_wgn"
leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                      "name": name,
                      "simple_name": "mcpilco",
                      "short_description": "Random wgn exploration.",
                      "readme_path": f"readmes/{name}.md",
                      "username": 'turcato-niccolo'}

torque_limit = [0.0, 6.0]
T_sym = 0.002

T_control = 0.02 # 50 Hz
# controller parameters
ctrl_rate = int(T_control/T_sym)
u_max = torque_limit[1]
n_dof = 2
controlled_joint = [1]

controller = Controller_Random_exploration(ctrl_rate=10, filt_freq=10, seed=0, type_random='SUM_SIN', controlled_dof=[1],
                                           random_par={'sin_freq': 10, 'num_sin': 5, 'sin_amp': 1.6, 'butter_order': 2},
                                           wait_steps=10)

controller.init()
