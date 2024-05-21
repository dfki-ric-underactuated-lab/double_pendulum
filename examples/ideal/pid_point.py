import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.utils.plotting import plot_timeseries

# model parameters
design = "design_A.0"
model = "model_1.0"
robot = "double_pendulum"
torque_limit = [5.0, 5.0]

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 4.0
integrator = "runge_kutta"
x0 = [np.pi - 0.5, 0.2, 0.0, 0.2]
goal = [np.pi, 0.0, 0.0, 0.0]

# controller parameters
Kp = 10.0
Ki = 0.0
Kd = 0.1

# setup savedir
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "point_pid", timestamp)
os.makedirs(save_dir)

# setup simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)

controller = PointPIDController(torque_limit=torque_limit, dt=dt)
controller.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller.set_goal(goal)
controller.init()
T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=x0,
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
    video_name=os.path.join(save_dir, "simulation"),
)

plot_timeseries(
    T,
    X,
    U,
    pos_y_lines=[0.0, np.pi],
    tau_y_lines=[-torque_limit[1], torque_limit[1]],
    save_to=os.path.join(save_dir, "time_series"),
)
