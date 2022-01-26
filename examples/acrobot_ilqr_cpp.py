import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.utils.plotting import plot_timeseries


# model parameters
mass = [0.608, 0.630]
length = [0.3, 0.2]
com = [0.275, 0.166]
damping = [0.081, 0.0]
# cfric = [0.093, 0.186]
cfric = [0., 0.]
gravity = 9.81
inertia = [0.05472, 0.2522]
torque_limit = [0.0, 6.0]

# simulation parameter
dt = 0.001
t_final = 20.0
integrator = "runge_kutta"

# controller parameters
N = 1000
N_init = 5000
max_iter = 100
regu_init = 100
break_cost_redu = 1e-6
sCu = [0.005, 0.005]
sCp = [0., 0.]
sCv = [0., 0.]
sCen = 0.
fCp = [1000., 1000.]
fCv = [10., 10.]
fCen = 0.

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

plant = SymbolicDoublePendulum(mass=mass,
                               length=length,
                               com=com,
                               damping=damping,
                               gravity=gravity,
                               coulomb_fric=cfric,
                               inertia=inertia,
                               torque_limit=torque_limit)

sim = Simulator(plant=plant)

controller = ILQRMPCCPPController(mass=mass,
                                  length=length,
                                  com=com,
                                  damping=damping,
                                  gravity=gravity,
                                  coulomb_fric=cfric,
                                  inertia=inertia,
                                  torque_limit=torque_limit)

controller.set_start(start)
controller.set_goal(goal)
controller.set_parameters(N=N,
                          dt=dt,
                          max_iter=1,
                          regu_init=regu_init,
                          break_cost_redu=break_cost_redu,
                          sCu=sCu,
                          sCp=sCp,
                          sCv=sCv,
                          sCen=sCen,
                          fCp=fCp,
                          fCv=fCv,
                          fCen=fCen,
                          integrator=integrator)
controller.compute_init_traj(N=N_init,
                             dt=dt,
                             max_iter=max_iter,
                             regu_init=regu_init,
                             break_cost_redu=break_cost_redu,
                             sCu=sCu,
                             sCp=sCp,
                             sCv=sCv,
                             sCen=sCen,
                             fCp=fCp,
                             fCv=fCv,
                             fCen=fCen,
                             integrator=integrator)
controller.init()
T, X, U = sim.simulate_and_animate(t0=0.0, x0=start,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta", phase_plot=False)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]])

