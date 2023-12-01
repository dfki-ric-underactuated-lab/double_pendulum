import time
from datetime import datetime
import os
import numpy as np

from double_pendulum.trajectory_optimization.direct_collocation.direct_collocation import DirCol
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.trajectory_following.trajectory_controller import TrajectoryController


# model parameters
design = "design_A.0"
model = "model_1.2"
robot = "acrobot"

urdf_path = "../../data/urdfs/design_A.0/model_1.0/"+robot+".urdf"

torque_limit = 6.0

mpar = model_parameters(
    torque_limit = [torque_limit]*2,
    model_design=design,
    model_id=model,
    robot=robot
    )


# Trajectory parameters
initial_state = np.array([0.0, 0.0, 0., 0.])
final_state = np.array([np.pi, 0.0, 0.0, 0.0])
n = 20
init_traj_time_interval = [0., 10.]
freq = 1000

# limits
theta_limit = float(np.deg2rad(360.))
speed_limit = 10
minimum_timestep = 0.01
maximum_timestep = 0.2

# Initial guesses
X_initial = np.zeros((4,n))
X_initial[0,:] = np.linspace(start=0,stop=np.pi, num=n)
U_initial = np.zeros((2,n))
h_initial = maximum_timestep

# costs
R = np.eye(2) * 0.01 
Q = np.diag([1,1,20,20])

wh = 20
# saving
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data/trajectories", design, model, robot, "dircol", "trajopt", timestamp)
os.makedirs(save_dir)


# Direct Collocation calculation
t0 = time.time()
dc = DirCol(urdf_path,
        robot,
        modelPars=mpar,
        saveDir=save_dir)
dc.MathematicalProgram(
    N=n,
    wh=wh,
    R=R,
    Q=Q,
    h_min=minimum_timestep,
    h_max=maximum_timestep,
    x0=initial_state,
    xf=final_state,
    torque_limit=torque_limit,
    X_initial=X_initial,
    U_initial=U_initial,
    h_initial=h_initial
)

T, X, U = dc.ComputeTrajectory(freq=freq)
print("Computing time: ", time.time() - t0, "s")
traj_file = os.path.join(save_dir, "trajectory.csv")
save_trajectory(csv_path=traj_file,
                T=T, X=X, U=U)
# plotting
plot_timeseries(T, X, U, None,
                plot_energy=False,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit, torque_limit],
                save_to=os.path.join(save_dir, "timeseries"))

## animate TODO: currently not supported
## animation with meshcat in browser window
#print("animating...")
#dc.animate_trajectory()
#
## simulate in python plant
#dt = T[1] - T[0]
#t_final = T[-1]
#x0 = X[0]
#
#plant = SymbolicDoublePendulum(model_pars=mpar)
#sim = Simulator(plant=plant)
#
#controller = TrajectoryController(csv_path=traj_file,
#                                  torque_limit=torque_limit,
#                                  kK_stabilization=False)
#controller.init()
#
#T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
#                                   tf=t_final, dt=dt, controller=controller,
#                                   integrator="runge_kutta",
#                                   plot_inittraj=True)
#
#plot_timeseries(T, X, U, None,
#                plot_energy=False,
#                pos_y_lines=[0.0, np.pi],
#                tau_y_lines=[-torque_limit_active, torque_limit_active])
#print("done")