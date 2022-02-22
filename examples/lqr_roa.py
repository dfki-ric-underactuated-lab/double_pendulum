import os
from datetime import datetime
import yaml
import numpy as np

from roatools.obj_fcts import caprr_coopt_interface
from roatools.vis import plotEllipse


# model parameters
robot = "acrobot"

# real parameters
# mass = [0.608, 0.630]
# length = [0.3, 0.2]
# com = [0.275, 0.166]
# # damping = [0.081, 0.0]
# damping = [0.0, 0.0]
# # cfric = [0.093, 0.186]
# cfric = [0., 0.]
# gravity = 9.81
# inertia = [0.05472, 0.02522]
# if robot == "acrobot":
#     torque_limit = [0.0, 5.0]
# if robot == "pendubot":
#     torque_limit = [5.0, 0.0]

# design optimized parameters
mass = [0.608, 0.632]
length = [0.40, 0.23]
com = [length[0], length[1]]
# damping = [0.081, 0.0]
damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
if robot == "acrobot":
    torque_limit = [0.0, 5.0]
if robot == "pendubot":
    torque_limit = [5.0, 0.0]
# simulation parameter
# dt = 0.005
# t_final = 6.0
# integrator = "runge_kutta"

# motion parameters
goal = [np.pi, 0, 0, 0]

# roa parameters
roa_backend = "sos"
# controller parameters
if robot == "acrobot":
    Q = np.diag((6.61, 1.59, 0.02, 0.11))
    R = np.diag((0.09, 0.09))
if robot == "pendubot":
    Q = np.diag((87.40, 9.60, 0.012, 0.090))
    R = np.diag((0.24, 0.24))

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "lqr", "roa", timestamp)
os.makedirs(save_dir)

design_params = {"m": mass,
                 "l": length,
                 "lc": com,
                 "b": damping,
                 "fc": cfric,
                 "g": gravity,
                 "I": inertia,
                 "tau_max": torque_limit}

roa_calc = caprr_coopt_interface(design_params=design_params,
                                 Q=Q,
                                 R=R,
                                 backend=roa_backend)
roa_calc._update_lqr(Q=Q, R=R)
vol, rho_f, S = roa_calc._estimate()

np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
np.savetxt(os.path.join(save_dir, "vol"), [vol])
# np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)

plotEllipse(goal[0], goal[1], 0, 1, rho_f, S,
            save_to=os.path.join(save_dir, "roaplot"),
            show=True)

np.savetxt(os.path.join(save_dir, "controller_par.csv"),
           [Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3], R[0, 0]])

par_dict = {"mass1": mass[0],
            "mass2": mass[1],
            "length1": length[0],
            "length2": length[1],
            "com1": com[0],
            "com2": com[1],
            "inertia1": inertia[0],
            "inertia2": inertia[1],
            "damping1": damping[0],
            "damping2": damping[1],
            "coulomb_friction1": cfric[0],
            "coulomb_friction2": cfric[1],
            "gravity": gravity,
            "torque_limit1": torque_limit[0],
            "torque_limit2": torque_limit[1],
            # "dt": dt,
            # "t_final": t_final,
            # "integrator": integrator,
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3]
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
