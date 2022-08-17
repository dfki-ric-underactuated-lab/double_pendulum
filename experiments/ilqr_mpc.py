import numpy as np

from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment as run_experiment_t
#from double_pendulum.experiments.hardware_control_loop_mjbots import run_experiment as run_experiment_mj


robot = "acrobot"
motors = "tmotors"

# model parameters
mass = [0.608, 0.5]
length = [0.3, 0.4]
com = [length[0], length[1]]
# damping = [0.081, 0.0]
damping = [0.0, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [mass[0]*length[0]**2, mass[1]*length[1]**2]
if robot == "acrobot":
    torque_limit = [0.0, 6.0]
if robot == "pendubot":
    torque_limit = [4.0, 0.0]

# simulation parameter
dt = 0.005
t_final = 5.0
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

# init trajectory
init_csv_path = "trajectory.csv"

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# construct controller
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
controller.init()

if motors == "tmotors":
    run_experiment_t(controller=controller,
                     dt=dt,
                     t_final=t_final,
                     can_port="can0",
                     motor_ids=[8, 9],
                     tau_limit=torque_limit,
                     friction_compensation=True,
                     #friction_terms=[0.093, 0.081, 0.186, 0.0],
                     friction_terms=[0.093, 0.005, 0.15, 0.001],
                     velocity_filter="lowpass",
                     filter_args={"alpha": 0.2,
                                  "kernel_size": 5,
                                  "filter_size": 1},
                     save_dir="data/acrobot/tmotors/ilqr_results/mpc")
# elif motors == "mjbots":
#     asyncio.run(run_experiment_mj(controller=controller,
#                                   dt=dt,
#                                   t_final=t_final,
#                                   motor_ids=[1, 2],
#                                   tau_limit=torque_limit,
#                                   friction_compensation=False,
#                                   friction_terms=[0.0, 0.0, 0.0, 0.0],
#                                   velocity_filter="lowpass",
#                                   filter_args={"alpha": 0.15,
#                                                "kernel_size": 21,
#                                                "filter_size": 21},
#                                   save_dir="data/acrobot/mjbots/ilqr_results/mpc"))
