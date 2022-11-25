import os
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.controller.lqr.roa.ellipsoid import plotEllipse


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
#dpar = [0.10371484, 0.20015142, 0.39745556]
#dpar = [0.2146793, 0.20166482, 0.39985249]
#dpar = [0.1000112, 0.20000669, 0.39371824]
#dpar = [0.20512917, 0.20007062, 0.39969301]
#dpar = [0.22050581, 0.2003057, 0.39906214]
dpar = [0.10036469, 0.20000136, 0.39834337]
mass = [0.608, dpar[0]]
length = [dpar[1], dpar[2]]
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
roa_backend = "najafi"
najafi_evals = 100000
# controller parameters
if robot == "acrobot":
    #cpar = [0.99589986, 0.72042251, 0.01029466, 0.05514445, 0.56246704]
    #cpar = [9.92722271e+00, 5.90250477e+00, 9.84836109e-01, 5.65253103e-01, 1.06118093e-03]
    #cpar = [1.53358595, 9.89761653, 0.23111711, 0.18065673, 0.95891982]
    #cpar = [7.56822705, 1.85180584, 0.99891571, 0.99944761, 0.75677003]
    #cpar = [2.07884366, 0.15380351, 0.98670823, 0.99673571, 0.61940116]
    cpar = [0.10683852, 7.99928943, 0.18692845, 0.14953787, 0.74022557]
    Q = np.diag((cpar[0], cpar[1], cpar[2], cpar[3]))
    R = np.diag((cpar[4], cpar[4]))
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
                                 backend=roa_backend,
                                 najafi_evals=najafi_evals)
roa_calc._update_lqr(Q=Q, R=R)
vol, rho_f, S = roa_calc._estimate()

print("rho: ", rho_f)
print("volume: ", vol)
np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
np.savetxt(os.path.join(save_dir, "vol"), [vol])
# np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)
np.savetxt(os.path.join(save_dir, "Smatrix"), S)

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
