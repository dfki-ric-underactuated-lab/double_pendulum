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
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_top


# model parameters
robot = "acrobot"
with_lqr = True

mass = [1.0, 1.0]
length = [1.0, 2.0]
com = [0.5, 1.0]
damping = [0.0, 0.0]
cfric = [0.0, 0.0]
gravity = 9.8
inertia = [mass[0]*com[0]**2+0.083, mass[1]*com[1]**2+0.33]
motor_inertia = 0.
torque_limit = [0.0, 25.0]

# simulation parameters
# results significantly depend on the integration method and time step!
integrator = "euler"
goal = [np.pi, 0., 0., 0.]
dt = 0.0045
x0 = [np.pi/2.-1.4, 0.0, 0.0, 0.0]
t_final = 15.0

# controller parameters
kp = 61.2  # > 61.141
kd = 35.8  # > 35.741
kv = 66.3  # > 0.0

Q = np.eye(4)
R = np.eye(2)


def condition1(t, x):
    return False

def condition2(t, x):
    goal = [np.pi, 0., 0., 0.]
    zeta = 0.08  # 0.04

    y = wrap_angles_top(x)

    delta = np.abs(y - goal)
    delta[2] *= 0.1
    delta[3] *= 0.1
    s = np.sum(delta)
    return s < zeta


plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               motor_inertia=motor_inertia,
                               torque_limit=torque_limit)
sim = Simulator(plant=plant)

if with_lqr:
    controller1 = EnergyController(mass=mass,
                                   length=length,
                                   com=com,
                                   damping=damping,
                                   gravity=gravity,
                                   coulomb_fric=cfric,
                                   inertia=inertia,
                                   motor_inertia=motor_inertia,
                                   torque_limit=torque_limit)
    controller1.set_parameters(kp=kp, kd=kd, kv=kv)
    controller1.set_goal(goal)

    controller2 = LQRController(mass=mass,
                                length=length,
                                com=com,
                                damping=damping,
                                gravity=gravity,
                                coulomb_fric=cfric,
                                inertia=inertia,
                                torque_limit=torque_limit)
    controller2.set_goal(goal)
    controller2.set_cost_matrices(Q=Q, R=R)
    controller2.set_parameters(failure_value=0.0,
                               cost_to_go_cut=1e9)

    controller = CombinedController(
            controller1=controller1,
            controller2=controller2,
            condition1=condition1,
            condition2=condition2)
    controller.init()
    print(f"LQR controller K matrix: {controller2.K}")
else:
    controller = EnergyController(mass=mass,
                                  length=length,
                                  com=com,
                                  damping=damping,
                                  gravity=gravity,
                                  coulomb_fric=cfric,
                                  inertia=inertia,
                                  motor_inertia=motor_inertia,
                                  torque_limit=torque_limit)
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
if with_lqr:
    energy = controller1.en
    des_energy = controller1.desired_energy
else:
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
                tau_y_lines=[-torque_limit[1], torque_limit[1]],
                energy_y_lines=[des_energy],
                save_to=os.path.join(save_dir, "time_series"))

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
