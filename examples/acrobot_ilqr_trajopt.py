import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.trajectory_optimization.ilqr.ilqr_cpp import ilqr_calculator
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
t_final = 5.0
integrator = "runge_kutta"

# controller parameters
N = 5000
max_iter = 1000
regu_init = 100
break_cost_redu = 1e-6
sCu = [1e-4, 1e-4]
sCp = [0., 0.]
sCv = [0., 0.]
sCen = 0.
fCp = [1e3, 1e3]
fCv = [10., 10.]
fCen = 0.

# swingup parameters
start = [0., 0., 0., 0.]
goal = [np.pi, 0., 0., 0.]

il = ilqr_calculator()
il.set_model_parameters(mass=mass,
                        length=length,
                        com=com,
                        damping=damping,
                        gravity=gravity,
                        coulomb_fric=cfric,
                        inertia=inertia,
                        torque_limit=torque_limit)
il.set_parameters(N=N,
                  dt=dt,
                  max_iter=max_iter,
                  regu_init=regu_init,
                  break_cost_redu=break_cost_redu,
                  integrator=integrator)
il.set_cost_parameters(sCu=sCu,
                       sCp=sCp,
                       sCv=sCv,
                       sCen=sCen,
                       fCp=fCp,
                       fCv=fCv,
                       fCen=fCen)
il.set_start(start)
il.set_goal(goal)

T, X, U = il.compute_trajectory()

U = np.append(U, [[0.0, 0.0]], axis=0)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[1], torque_limit[1]])
