import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import load_trajectory, trajectory_properties


# model parameters
design = "design_A.0"
model = "model_2.0"

model_par_path = (
    "../../data/system_identification/identified_parameters/"
    + design
    + "/"
    + model
    + "/model_parameters.yml"
)
mpar = model_parameters()
mpar.load_yaml(model_par_path)
# mpar.set_damping([0., 0.])
# mpar.set_cfric([0., 0.])

# trajectory
csv_path = (
    "../../data/system_identification/excitation_trajectories/trajectory-pos-50.csv"
)
with_tau = False
num_break = 250

# simulation parameters
T_des, X_des, U_des = load_trajectory(csv_path, with_tau=with_tau)
dt, t_final, x0, _ = trajectory_properties(T_des, X_des)
integrator = "runge_kutta"

# controller parameters
Kp = 100.0
Ki = 0.0
Kd = 1.0

# simulation objects
plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

controller = TrajPIDController(
    csv_path=csv_path,
    use_feed_forward_torque=with_tau,
    torque_limit=mpar.tl,
    num_break=num_break,
)

controller.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)
controller.init()

T, X, U = sim.simulate_and_animate(
    t0=0.0,
    x0=x0,
    tf=t_final,
    dt=dt,
    controller=controller,
    integrator=integrator,
    plot_inittraj=True,
    save_video=True,
    anim_dt=0.02,
)

# T, X, U = sim.simulate(t0=0.0, x0=x0,
#                        tf=t_final, dt=dt, controller=controller,
#                        integrator=integrator)

plot_timeseries(T, X, U, pos_y_lines=[-np.pi, 0.0, np.pi], T_des=T_des, X_des=X_des)
