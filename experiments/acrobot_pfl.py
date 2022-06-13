import numpy as np

from double_pendulum.controller.partial_feedback_linearization.symbolic_pfl import SymbolicPFLAndLQRController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment


robot = "acrobot"
pfl_method = "noncollocated"

mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
cfric = [0.093, 0.186]
#cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
if robot == "acrobot":
    torque_limit = [0.0, 3.0]
if robot == "pendubot":
    torque_limit = [3.0, 0.0]

dt = 0.005
t_final = 5.0

if robot == "acrobot":
    if pfl_method == "collocated":
        # par = [9.94271982, 1.56306923, 3.27636175]  # ok in sim with cfric
        # par = [9.94246152, 9.84124115, 9.81120166]  # good in sim without cfric
        par = [7.94246152, 5.84124115, 5.81120166]
    elif pfl_method == "noncollocated":
        par = [9.19534629, 2.24529733, 5.90567362]  # good
    else:
        print(f"pfl_method {pfl_method} not found. Please set eigher" +
              "pfl_method='collocated' or pfl_method='noncollocated'")
        exit()
elif robot == "pendubot":
    if pfl_method == "collocated":
        par = [6.97474837, 9.84031538, 9.1297417]  # bad
    elif pfl_method == "noncollocated":
        # par = [6.97474837, 9.84031538, 9.1297417]  # bad
        # par = [4.91129641, 10., 1.64418209]
        par = [26.34039456, 99.99876263, 11.89097532]
        # par = [15.64747394, 19.5291726, 3.71447987]
    else:
        print(f"pfl_method {pfl_method} not found. Please set eigher" +
              "pfl_method='collocated' or pfl_method='noncollocated'")
        exit()
else:
    print(f"robot {robot} not found. Please set eigher" +
          "robot='acrobot' or robot='pendubot'")
    exit()


controller = SymbolicPFLAndLQRController(mass,
                                         length,
                                         com,
                                         damping,
                                         gravity,
                                         cfric,
                                         inertia,
                                         torque_limit,
                                         robot,
                                         pfl_method)

controller.set_goal([np.pi, 0., 0., 0.])
controller.en_controller.set_cost_parameters(kpos=par[0],
                                             kvel=par[1],
                                             ken=par[2])
controller.init()

run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[8, 9],
               tau_limit=torque_limit,
               friction_compensation=True,
               friction_terms=[0.093, 0.186, 0.081, 0.0],
               velocity_filter="lowpass",
               filter_args={"alpha": 0.2,
                            "kernel_size": 5,
                            "filter_size": 1},
               save_dir="data/"+robot+"/tmotors/pfl_results")
