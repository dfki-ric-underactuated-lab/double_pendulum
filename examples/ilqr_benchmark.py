import os
from datetime import datetime
import numpy as np
import pickle

from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.analysis.benchmark import benchmarker
from double_pendulum.analysis.utils import get_par_list

robot = "acrobot"

# model parameters
mass = [0.608, 0.63]
length = [0.3, 0.4]
com = [length[0], length[1]]
com = [0.275, 0.166]
# damping = [0.081, 0.0]
damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
motor_inertia = 8.8e-5
if robot == "acrobot":
    torque_limit = [0.0, 4.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

# simulation parameter
dt = 0.005
t_final = 4.99
integrator = "runge_kutta"

# controller parameters
N = 100
N_init = 1000
max_iter = 5
max_iter_init = 1000
regu_init = 100
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6

# acrobot good par
sCu = [9.97938814e-02, 9.97938814e-02]
sCp = [2.06969312e-02, 7.69967729e-02]
sCv = [1.55726136e-04, 5.42226523e-03]
sCen = 0.0
fCp = [3.82623819e+02, 7.05315590e+03]
fCv = [5.89790058e+01, 9.01459500e+01]
fCen = 0.0

Q = np.array([[sCp[0], 0., 0., 0.],
              [0., sCp[1], 0., 0.],
              [0., 0., sCv[0], 0.],
              [0., 0., 0., sCv[1]]])
Qf = np.array([[fCp[0], 0., 0., 0.],
               [0., fCp[1], 0., 0.],
               [0., 0., fCv[0], 0.],
               [0., 0., 0., fCv[1]]])
R = np.array([[sCu[0], 0.],
              [0., sCu[1]]])

# benchmark parameters
mpar_vars = ["Ir",
             "m1r1", "I1", "b1", "cf1",
             "m2r2", "m2", "I2", "b2", "cf2"]

Ir_var_list = [0.0, motor_inertia, 2*motor_inertia]
m1r1_var_list = get_par_list(mass[0]*com[0], 0.5, 1.5, 3)
I1_var_list = get_par_list(inertia[0], 0.5, 1.5, 3)
b1_var_list = [0.0, 0.081, 0.19]
cf1_var_list = [0.0, 0.093, 0.186]
m2r2_var_list = get_par_list(mass[1]*com[1], 0.5, 1.5, 3)
m2_var_list = get_par_list(mass[1], 0.5, 1.5, 3)
I2_var_list = get_par_list(inertia[1], 0.5, 1.5, 3)
b2_var_list = [0.0, 0.081, 0.19]
cf2_var_list = [0.0, 0.093, 0.186]

modelpar_var_lists = {"Ir": Ir_var_list,
                      "m1r1": m1r1_var_list,
                      "I1": I1_var_list,
                      "b1": b1_var_list,
                      "cf1": cf1_var_list,
                      "m2r2": m2r2_var_list,
                      "m2": m2_var_list,
                      "I2": I2_var_list,
                      "b2": b2_var_list,
                      "cf2": cf2_var_list}


# init trajectory
latest_dir = sorted(os.listdir(os.path.join("data", robot, "ilqr", "trajopt")))[-1]
init_csv_path = os.path.join("data", robot, "ilqr", "trajopt", latest_dir, "trajectory.csv")

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# create save directory
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "ilqr", "mpc_benchmark", timestamp)
os.makedirs(save_dir)

# construct simulation objects
controller = ILQRMPCCPPController(mass=mass,
                                  length=length,
                                  com=com,
                                  damping=damping,
                                  gravity=gravity,
                                  coulomb_fric=cfric,
                                  inertia=inertia,
                                  torque_limit=torque_limit)

controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(N=N,
                          dt=dt,
                          max_iter=max_iter,
                          regu_init=regu_init,
                          max_regu=max_regu,
                          min_regu=min_regu,
                          break_cost_redu=break_cost_redu,
                          integrator=integrator)
controller.set_cost_parameters(sCu=sCu,
                               sCp=sCp,
                               sCv=sCv,
                               sCen=sCen,
                               fCp=fCp,
                               fCv=fCv,
                               fCen=fCen)
controller.load_init_traj(csv_path=init_csv_path)

ben = benchmarker(controller=controller,
                  x0=start,
                  dt=dt,
                  t_final=t_final,
                  goal=goal,
                  integrator=integrator,
                  save_dir=save_dir)
ben.set_model_parameter(mass=mass,
                        length=length,
                        com=com,
                        damping=damping,
                        gravity=gravity,
                        cfric=cfric,
                        inertia=inertia,
                        motor_inertia=motor_inertia,
                        torque_limit=torque_limit)
ben.set_init_traj(init_csv_path, read_with="numpy")
ben.set_cost_par(Q=Q, R=R, Qf=Qf)
ben.compute_ref_cost()
res = ben.benchmark(compute_model_robustness=True,
                    mpar_vars=mpar_vars,
                    modelpar_var_lists=modelpar_var_lists)
print(res)
f = open(os.path.join(save_dir, "results.pkl"), 'wb')
pickle.dump(res, f)
f.close()
