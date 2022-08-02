import os
from datetime import datetime
import yaml
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.controller.energy.energy_Xin import EnergyController

# model parameters
robot = "acrobot"

motor_inertia = 0.
damping = [0.0, 0.0]
cfric = [0.0, 0.0]
torque_limit = [0.0, 5.0]
active_act = 1

model_par_path = "../data/system_identification/identified_parameters/tmotors_v1.0/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
mpar.set_damping(damping)
mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

# simulation parameters
integrator = "runge_kutta"
goal = [np.pi, 0., 0., 0.]
dt = 0.002
x0 = [0.1, 0.0, 0.0, 0.0]
t_final = 60.0

# controller parameters
kp = 0.68  # > 0.67
kd = 0.3  # > 0.022
kv = 5.0  # > 0.0

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = EnergyController(model_pars=mpar)
controller.set_parameters(kp=kp, kd=kd, kv=kv)
controller.set_goal(goal)
# controller.check_parameters()
controller.init()

T, X, U = sim.simulate_and_animate(t0=0.0,
                                   x0=x0,
                                   tf=t_final,
                                   dt=dt,
                                   controller=controller,
                                   integrator=integrator,
                                   phase_plot=False,
                                   save_video=False)

# controller.save(path)
energy = controller.en
des_energy = controller.desired_energy

# saving and plotting
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", robot, "energy_Xin", timestamp)
os.makedirs(save_dir)

save_trajectory(csv_path=os.path.join(save_dir, "trajectory.csv"),
                T=T,
                X=X,
                U=U)

plot_timeseries(T=T, X=X, U=U, energy=energy,
                plot_energy=True,
                pos_y_lines=[-np.pi, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                energy_y_lines=[des_energy],
                save_to=os.path.join(save_dir, "time_series"))

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))

par_dict = {
            "dt": dt,
            "t_final": t_final,
            "integrator": integrator,
            "start_pos1": x0[0],
            "start_pos2": x0[1],
            "start_vel1": x0[2],
            "start_vel2": x0[3],
            "goal_pos1": goal[0],
            "goal_pos2": goal[1],
            "goal_vel1": goal[2],
            "goal_vel2": goal[3],
            }

with open(os.path.join(save_dir, "parameters.yml"), 'w') as f:
    yaml.dump(par_dict, f)
