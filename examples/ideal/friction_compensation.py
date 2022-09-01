import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.friction_compensation.friction_compensation_controller import FrictionCompensationController
from double_pendulum.utils.plotting import plot_timeseries


design = "design_A.0"
model = "model_1.0"
robot = "double_pendulum"

torque_limit = [10.0, 10.0]

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"

mpar = model_parameters()
mpar.load_yaml(model_par_path)
#mpar.set_motor_inertia(0.)
#mpar.set_damping([0., 0.])
#mpar.set_cfric([0., 0.])
mpar.set_torque_limit(torque_limit)

# simulation parameters
dt = 0.002
t_final = 10.0
integrator = "runge_kutta"
x0 = [np.pi/2., 0., 0., 0.]

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "friction_compensation", timestamp)
os.makedirs(save_dir)

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = FrictionCompensationController(model_pars=mpar)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator=integrator,
                                   save_video=False,
                                   video_name=os.path.join(save_dir, "simulation"))
X_meas = sim.meas_x_values
X_filt = sim.filt_x_values
U_con = sim.con_u_values

plot_timeseries(T, X, U, None,
                plot_energy=False,
                X_meas=X_meas,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[0], torque_limit[0]],
                save_to=os.path.join(save_dir, "time_series"))
