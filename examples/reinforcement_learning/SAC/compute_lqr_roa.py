import os
import yaml
import numpy as np

from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.controller.lqr.roa.ellipsoid import plotEllipse


# model parameters
robot = "acrobot"

if robot == "acrobot":
    design = "design_C.0"
    model = "model_3.0"
    torque_limit = [0.0, 5.0]
if robot == "pendubot":
    design = "design_A.0"
    model = "model_2.0"
    torque_limit = [5.0, 0.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters(filepath=model_par_path)

mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# motion parameters
goal = [np.pi, 0, 0, 0]

# roa parameters
roa_backend = "sos_con"
najafi_evals = 100000

# controller parameters
if robot == "acrobot":
    # Q = np.diag([0.0125, 6.5, 6.88, 9.36])
    # R = np.diag([2.4, 2.4])
    # Q = np.diag([1.0, 1.0, 1.0, 1.0])
    # R = np.diag([1.0, 1.0])
    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))
if robot == "pendubot":
    # Q = np.diag([0.00125, 0.65, 0.000688, 0.000936])
    # R = np.diag([25.0, 25.0])
    # Q = np.diag([1.0, 1.0, 1.0, 1.0])
    # R = np.diag([1.0, 1.0])
    # Q = np.diag([0.0125, 6.5, 6.88, 9.36])
    # R = np.diag([2.4, 2.4])
    #Q = 1.25*np.diag([0.64, 0.56, 0.15, 0.14])
    Q = 3.0*np.diag([0.64, 0.64, 0.1, 0.1])
    R = np.eye(2)*0.82


# savedir
save_dir = os.path.join("lqr_data", robot, "lqr", "roa")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

design_params = {
    "m": mpar.m,
    "l": mpar.l,
    "lc": mpar.r,
    "b": mpar.b,
    "fc": mpar.cf,
    "g": mpar.g,
    "I": mpar.I,
    "tau_max": mpar.tl,
}

roa_calc = caprr_coopt_interface(
    design_params=design_params,
    Q=Q,
    R=R,
    backend=roa_backend,
    najafi_evals=najafi_evals,
    robot=robot,
    verbose=False,
)
roa_calc._update_lqr(Q=Q, R=R)
vol, rho_f, S = roa_calc._estimate()

print(robot)
print("Q=", Q)
print("R=", R)
print("rho: ", rho_f)
print("volume: ", vol)
print("S", S)
np.savetxt(os.path.join(save_dir, "rho"), [rho_f])
np.savetxt(os.path.join(save_dir, "vol"), [vol])
# np.savetxt(os.path.join(save_dir, "rhohist"), rhoHist)
np.savetxt(os.path.join(save_dir, "Smatrix"), S)

plotEllipse(
    goal[0],
    goal[1],
    0,
    1,
    rho_f,
    S,
    save_to=os.path.join(save_dir, "roaplot"),
    show=True,
)

np.savetxt(
    os.path.join(save_dir, "controller_par.csv"),
    [Q[0, 0], Q[1, 1], Q[2, 2], Q[3, 3], R[0, 0]],
)

mpar.save_dict(os.path.join(save_dir, "parameters.yml"))
