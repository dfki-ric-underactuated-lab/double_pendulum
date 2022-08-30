import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.gravity_compensation.PID_gravity_compensation_controller import PIDGravityCompensationController
from double_pendulum.utils.plotting import plot_timeseries


robot = "double_pendulum"

torque_limit = [10.0, 10.0]

model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"

mpar = model_parameters()
mpar.load_yaml(model_par_path)
#mpar.set_motor_inertia(motor_inertia)
#mpar.set_damping(damping)
#mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"
x0 = [np.pi/2., np.pi/4., 0., 0.]
goal = [np.pi, 0., 0., 0.]

# controller parameters
Kp = 1.
Ki = 0.
Kd = 0.1

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "gravity_compensation_pd", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller = PIDGravityCompensationController(model_pars=mpar, dt=dt)
controller.set_parameters(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd)
controller.set_goal(goal)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"))
plot_timeseries(T, X, U,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[0], torque_limit[0]],
                save_to=os.path.join(save_dir, "time_series"))
