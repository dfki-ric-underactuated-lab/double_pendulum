import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller import TVLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

## model parameters
design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.)
mpar.set_damping([0., 0.])
mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

## trajectory parameters
csv_path = os.path.join("../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv")

## load reference trajectory
T_des, X_des, U_des = load_trajectory(csv_path)
#dt = T_des[1] - T_des[0]
t_final = T_des[-1]
dt = 0.002
#t_final = 10.
goal = [np.pi, 0., 0., 0.]

## simulation parameters
x0 = [0.0, 0.0, 0.0, 0.0]
integrator = "runge_kutta"

## controller parameters
if robot == "acrobot":
    #Q = np.diag([0.64, 0.56, 0.13, 0.067])
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = 0.001*np.eye(2)*0.82
elif robot == "pendubot":
    #Q = np.diag([0.64, 0.64, 0.4, 0.2])
    #R = np.eye(2)*0.82
    Q = 20.*np.diag([0.64, 0.64, 0.1, 0.1])
    R = np.eye(2)*0.82
Qf = np.copy(Q)
horizon = 1

## init plant, simulator and controller
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TVLQRController(
        model_pars=mpar,
        csv_path=csv_path,
        torque_limit=torque_limit,
        horizon=horizon)

controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller.init()

## simulate
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator,
                                   plot_inittraj=True)
## saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "tvlqr", timestamp)
os.makedirs(save_dir)

os.system(f"cp {csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

plot_timeseries(T, X, U,
                T_des=T_des,
                X_des=X_des,
                U_des=U_des,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "timeseries"))
