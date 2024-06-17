import time
from datetime import datetime
import os
import numpy as np
import yaml

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.trajectory_optimization.ilqr.ilqr_cpp import ilqr_calculator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.trajectory_following.trajectory_controller import (
    TrajectoryController,
)

## model parameters
design = "design_C.0"
model = "model_3.0"
robot = "acrobot"

if robot == "acrobot":
    torque_limit = [0.0, 5.0]
if robot == "pendubot":
    torque_limit = [5.0, 0.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)

# controller parameters
N = 6000
max_iter = 100000
regu_init = 100.0
max_regu = 1000000.0
min_regu = 0.01
break_cost_redu = 1e-6

# simulation parameter
dt = 0.001
# t_final = 5.0
t_final = N * dt
integrator = "runge_kutta"

if robot == "acrobot":
    # looking good
    # [9.97938814e-02 2.06969312e-02 7.69967729e-02 1.55726136e-04
    #  5.42226523e-03 3.82623819e+02 7.05315590e+03 5.89790058e+01
    #  9.01459500e+01]
    # sCu = [9.97938814e-02, 9.97938814e-02]
    # sCp = [2.06969312e-01, 7.69967729e-01]
    # sCv = [1.55726136e-03, 5.42226523e-02]
    # sCen = 0.0
    # fCp = [3.82623819e+03, 7.05315590e+04]
    # fCv = [5.89790058e+01, 9.01459500e+01]
    # fCen = 0.0

    # # very good
    # sCu = [9.97938814e-02, 9.97938814e-02]
    # sCp = [2.06969312e-02, 7.69967729e-02]
    # sCv = [1.55726136e-04, 5.42226523e-03]
    # sCen = 0.0
    # fCp = [3.82623819e+02, 7.05315590e+03]
    # fCv = [5.89790058e+01, 9.01459500e+01]
    # fCen = 0.0
    sCu = [9.97938814e01, 9.97938814e01]
    sCp = [2.06969312e01, 7.69967729e01]
    sCv = [1.55726136e-01, 5.42226523e-00]
    sCen = 0.0
    fCp = [3.82623819e02, 7.05315590e03]
    fCv = [5.89790058e01, 9.01459500e01]
    fCen = 0.0

    # sCu = [100., 100.]
    # sCp = [0.1, 0.1]
    # sCv = [0.5, 0.5]
    # sCen = 0.
    # fCp = [50000., 50000.]
    # fCv = [2000., 2000.]
    # fCen = 0.

    # stage_prefac = 0.1
    # final_prefac = 1.
    # sCu = [stage_prefac*.11, stage_prefac*.11]
    # sCp = [stage_prefac*.97, stage_prefac*.93]
    # sCv = [stage_prefac*.39, stage_prefac*.26]
    # sCen = 0.
    # fCp = [final_prefac*.97, final_prefac*.93]
    # fCv = [final_prefac*.39, final_prefac*.26]
    # fCen = 0.

    # sCu = [0.8220356078430472, 0.8220356078430472]
    # sCp = [0.6406768243361961, 0.5566465602921646]
    # sCv = [0.13170941522322516, 0.036794663247905396]
    # sCen = 0.
    # fCp = [0.7170451397596873, 0.7389953240562843]
    # fCv = [0.5243681881323512, 0.39819013775238776]
    # fCen = 0.

    # tvlqr parameters
    # u_prefac = 1.0
    # stage_prefac = 1.
    # final_prefac = 1.
    # sCu = [u_prefac*0.82, u_prefac*0.82]
    # sCp = [stage_prefac*0.64, stage_prefac*0.56]
    # sCv = [stage_prefac*0.13, stage_prefac*0.037]
    # sCen = 0.
    # fCp = [final_prefac*0.64, final_prefac*0.56]
    # fCv = [final_prefac*0.13, final_prefac*0.037]
    # fCen = 0.

    # [8.26303186e+01 2.64981012e+01 3.90215591e+01 3.87432205e+00
    #  2.47715889e+00 5.72238144e+04 9.99737172e+04 7.16184205e+03
    #  2.94688061e+03]
    # sCu = [89., 89.]
    # sCp = [40., 0.2]
    # sCv = [11., 1.0]
    # sCen = 0.0
    # fCp = [66000., 210000.]
    # fCv = [55000., 92000.]
    # fCen = 0.0

    # sCu = [89.53168298604868, 89.53168298604868]
    # sCp = [39.95840603845028, 0.220281011195961]
    # sCv = [10.853380829038803, 0.9882211066793491]
    # sCen = 0.
    # fCp = [65596.70698843336, 208226.67812877183]
    # fCv = [54863.83385207141, 91745.39489510724]
    # fCen = 0.
    # looking good
    # [9.96090757e-02 2.55362809e-02 9.65397113e-02 2.17121720e-05
    #  6.80616778e-03 2.56167942e+02 7.31751057e+03 9.88563736e+01
    #  9.67149494e+01]
    # sCu = [9.96090757e-02, 9.96090757e-02]
    # sCp = [2.55362809e-02, 9.65397113e-02]
    # sCv = [2.17121720e-05, 6.80616778e-03]
    # sCen = 0.0
    # fCp = [2.56167942e+02, 7.31751057e+03]
    # fCv = [9.88563736e+01, 9.67149494e+01]
    # fCen = 0.0

    # sCu = [0.2, 0.2]
    # sCp = [0.1, 0.2]
    # sCv = [0.2, 0.2]
    # sCen = 0.
    # fCp = [1500., 500.]
    # fCv = [10., 10.]
    # fCen = 0.

    # [9.64008003e-04 3.69465206e-04 9.00160028e-04 8.52634075e-04
    #  3.62146682e-05 3.49079107e-04 1.08953921e-05 9.88671633e+03
    #  7.27311031e+03 6.75242351e+01 9.95354381e+01 6.36798375e+01]

    # [6.83275883e-01 8.98205799e-03 5.94690881e-04 2.60169706e-03
    # 4.84307636e-03 7.78152311e-03 2.69548072e+02 9.99254272e+03
    # 8.55215256e+02 2.50563565e+02 2.57191000e+01]
if robot == "pendubot":
    sCu = [1.0, 1.0]
    sCp = [0.1, 0.2]
    sCv = [0.0, 0.0]
    sCen = 0.0
    fCp = [10.0, 20.0]
    fCv = [10.0, 20.0]
    fCen = 0.0

# swingup parameters
start = [0.0, 0.0, 0.0, 0.0]
goal = [np.pi, 0.0, 0.0, 0.0]

t0 = time.time()
il = ilqr_calculator()
il.set_model_parameters(model_pars=mpar)
il.set_parameters(
    N=N,
    dt=dt,
    max_iter=max_iter,
    regu_init=regu_init,
    max_regu=max_regu,
    min_regu=min_regu,
    break_cost_redu=break_cost_redu,
    integrator=integrator,
)
il.set_cost_parameters(
    sCu=sCu, sCp=sCp, sCv=sCv, sCen=sCen, fCp=fCp, fCv=fCv, fCen=fCen
)
il.set_start(start)
il.set_goal(goal)

# computing the trajectory
T, X, U = il.compute_trajectory()
print("Computing time: ", time.time() - t0, "s")

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "ilqr", "trajopt", timestamp)
os.makedirs(save_dir)

traj_file = os.path.join(save_dir, "trajectory.csv")

il.save_trajectory_csv()
os.system("mv trajectory.csv " + traj_file)
# save_trajectory(csv_path=filename,
#                 T=T, X=X, U=U)

par_dict = {
    "dt": dt,
    "t_final": t_final,
    "integrator": integrator,
    "start_pos1": start[0],
    "start_pos2": start[1],
    "start_vel1": start[2],
    "start_vel2": start[3],
    "goal_pos1": goal[0],
    "goal_pos2": goal[1],
    "goal_vel1": goal[2],
    "goal_vel2": goal[3],
    "N": N,
    "max_iter": max_iter,
    "regu_init": regu_init,
    "max_regu": max_regu,
    "min_regu": min_regu,
    "break_cost_redu": break_cost_redu,
    "sCu1": sCu[0],
    "sCu2": sCu[1],
    "sCp1": sCp[0],
    "sCp2": sCp[1],
    "sCv1": sCv[0],
    "sCv2": sCv[1],
    "sCen": sCen,
    "fCp1": fCp[0],
    "fCp2": fCp[1],
    "fCv1": fCv[0],
    "fCv2": fCv[1],
    "fCen": fCen,
}

with open(os.path.join(save_dir, "parameters.yml"), "w") as f:
    yaml.dump(par_dict, f)

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

# plotting
U = np.append(U, [[0.0, 0.0]], axis=0)
plot_timeseries(
    T,
    X,
    U,
    None,
    plot_energy=False,
    pos_y_lines=[0.0, np.pi],
    tau_y_lines=[-torque_limit[1], torque_limit[1]],
    save_to=os.path.join(save_dir, "timeseries"),
)

# simulation
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TrajectoryController(
    csv_path=traj_file, torque_limit=torque_limit, kK_stabilization=True
)
controller.init()

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=start,
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    phase_plot=False,
    save_video=False,
    video_name=os.path.join(save_dir, "simulation"),
    plot_inittraj=True,
)
