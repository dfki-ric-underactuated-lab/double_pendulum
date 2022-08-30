import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

# model parameters
robot = "acrobot"
urdf_path = "../../data/urdfs/"+robot+".urdf"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1

model_par_path = "../../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters_new2.yml"
mpar = model_parameters(filepath=model_par_path)
#mpar.set_motor_inertia(0.0)
#mpar.set_damping([0., 0.])
#mpar.set_cfric([0., 0.])
#mpar.set_mass([mpar.m[0], 0.9*mpar.m[1]])
#mpar.set_inertia([0.95*mpar.I[0], mpar.I[1]])
#mpar.set_torque_limit(torque_limit)

mpar_con = model_parameters(filepath=model_par_path)
#mpar_con.set_motor_inertia(0.)
#mpar_con.set_damping([0., 0.])
#mpar_con.set_cfric([0., 0.])
mpar_con.set_cfric([np.pi/2*mpar.cf[0], np.pi/2*mpar.cf[1]])
#mpar_con.set_torque_limit(torque_limit)

# trajectory parameters
#csv_path = "../data/trajectories/acrobot/dircol/acrobot_tmotors_swingup_1000Hz.csv"  # tmotors v1.0
csv_path = "../../data/trajectories/"+robot+"/ilqr_v1.0_new2/trajectory.csv"  # tmotors v1.0

# load reference trajectory
T_des, X_des, U_des = load_trajectory(csv_path)
#dt = T_des[1] - T_des[0]
dt = 0.005
t_final = T_des[-1] #  + 5
goal = [np.pi, 0., 0., 0.]

# simulation parameters
x0 = [0.0, 0.0, 0.0, 0.0]
integrator = "runge_kutta"

# controller parameters
if robot == "acrobot":
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = np.eye(1)*0.82
elif robot == "pendubot":
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = np.eye(1)*0.82
Qf = np.copy(Q)

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "tvlqr_drake", timestamp)
os.makedirs(save_dir)

# init plant, simulator and controller
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TVLQRController(
        csv_path=csv_path,
        urdf_path=urdf_path,
        model_pars=mpar_con,
        torque_limit=torque_limit,
        robot=robot,
        save_dir=save_dir)
controller.set_cost_parameters(Q=Q, R=R, Qf=Qf)

controller.init()

# simulate
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=True)

# saving and plotting
os.system(f"cp {csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

plot_timeseries(T, X, U,
                T_des=T_des,
                X_des=X_des,
                U_des=U_des,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "timeseries"))
