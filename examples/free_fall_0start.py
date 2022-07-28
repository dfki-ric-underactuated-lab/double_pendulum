import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import load_trajectory, save_trajectory


# model parameters
torque_limit = [6.0, 6.0]
model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 15.0
integrator = "runge_kutta"
x0 = np.array([0., 0., 0., 0.])

# trajectory
t1_final = 5.0
N = int(t1_final / dt)
T_des = np.linspace(0, t1_final, N+1)
p1_des = np.linspace(0, -np.pi/2, N+1)
p2_des = np.linspace(0, -np.pi/2, N+1)
v1_des = np.diff(p1_des, append=p1_des[-1]) / dt
v2_des = np.diff(p2_des, append=p2_des[-1]) / dt
X_des = np.array([p1_des, p2_des, v1_des, v2_des]).T

# controller parameters
Kp = 50.
Ki = 0.
Kd = 1.


def condition1(t, x):
    return False


def condition2(t, x):
    return t > 5.0


# simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller1 = TrajPIDController(T=T_des,
                                X=X_des,
                                read_with="numpy",
                                use_feed_forward_torque=False,
                                torque_limit=torque_limit,
                                num_break=40)
controller1.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller1.init()

controller2 = TrajPIDController(T=T_des,
                                X=X_des,
                                read_with="numpy",
                                use_feed_forward_torque=False,
                                torque_limit=torque_limit,
                                num_break=40)
controller2.set_parameters(Kp=0., Ki=0., Kd=0.)
controller2.init()

controller = CombinedController(
        controller1=controller1,
        controller2=controller2,
        condition1=condition1,
        condition2=condition2)

T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator, plot_inittraj=True,
                                   save_video=False)

plot_timeseries(T, X, U, None,
                T_des=T_des,
                X_des=X_des)
