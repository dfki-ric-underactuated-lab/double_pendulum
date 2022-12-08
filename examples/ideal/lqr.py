import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.utils.plotting import plot_timeseries


# model parameters
design = "design_A.0"
model = "model_2.0"
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

# simulation parameters
dt = 0.002
t_final = 3.0
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]

if robot == "acrobot":
    # x0 = [np.pi+0.05, -0.2, 0.0, 0.0]
    x0 = [np.pi+0.1, -0.1, -.5, 0.5]

    Q = np.diag((0.97, 0.93, 0.39, 0.26))
    R = np.diag((0.11, 0.11))

elif robot == "pendubot":
    x0 = [np.pi-0.2, 0.3, 0., 0.]

    Q = np.diag([0.00125, 0.65, 0.000688, 0.000936])
    R = np.diag([25.0, 25.0])

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "lqr", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller = LQRController(model_pars=mpar)
controller.set_goal(goal)
controller.set_cost_matrices(Q=Q, R=R)
controller.set_parameters(failure_value=0.0,
                          cost_to_go_cut=1000)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"))

plot_timeseries(T, X, U,
                X_meas=sim.meas_x_values,
                pos_y_lines=[np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "timeseries"))
