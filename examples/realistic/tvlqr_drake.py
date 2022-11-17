import os
from datetime import datetime
import numpy as np

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.tvlqr.tvlqr_controller_drake import TVLQRController
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.controller.ilqr.ilqr_mpc_cpp import ILQRMPCCPPController
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.wrap_angles import wrap_angles_top
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory

# model parameters
design = "design_A.0"
model = "model_2.0"
traj_model = "model_2.1"
robot = "acrobot"
urdf_path = "../../data/urdfs/design_A.0/model_1.0/"+robot+".urdf"

friction_compensation = True
stabilization = "lqr"

if robot == "pendubot":
    torque_limit = [5.0, 0.0]
    active_act = 0
elif robot == "acrobot":
    torque_limit = [0.0, 5.0]
    active_act = 1
torque_limit_pid = [5.0, 5.0]

model_par_path = "../../data/system_identification/identified_parameters/"+design+"/"+model+"/model_parameters.yml"
mpar = model_parameters(filepath=model_par_path)
mpar_con = model_parameters(filepath=model_par_path)
mpar_con.set_motor_inertia(0.)
if friction_compensation:
    mpar_con.set_damping([0., 0.])
    mpar_con.set_cfric([0., 0.])
mpar_con.set_torque_limit(torque_limit)

# trajectory parameters
csv_path = os.path.join("../../data/trajectories", design, traj_model, robot, "ilqr_1/trajectory.csv")

# load reference trajectory
T_des, X_des, U_des = load_trajectory(csv_path)
dt = T_des[1] - T_des[0]
t_final = T_des[-1] + 5
goal = [np.pi, 0., 0., 0.]

# simulation parameters
x0 = [0.0, 0.0, 0.0, 0.0]
integrator = "runge_kutta"

# noise
process_noise_sigmas = [0.0, 0.0, 0.0, 0.0]
meas_noise_sigmas = [0.0, 0.0, 0.05, 0.05]
delay_mode = "None"
delay = 0.0
u_noise_sigmas = [0., 0.]
u_responsiveness = 1.0
perturbation_times = []
perturbation_taus = []

# filter args
meas_noise_vfilter = "lowpass"
meas_noise_cut = 0.1
filter_kwargs = {"lowpass_alpha": [1., 1., 0.2, 0.2],
                 "kalman_xlin": goal,
                 "kalman_ulin": [0., 0.],
                 "kalman_process_noise_sigmas": process_noise_sigmas,
                 "kalman_meas_noise_sigmas": meas_noise_sigmas,
                 "ukalman_integrator": integrator,
                 "ukalman_process_noise_sigmas": process_noise_sigmas,
                 "ukalman_meas_noise_sigmas": meas_noise_sigmas}

# controller parameters
if robot == "acrobot":
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = np.eye(1)*0.82
elif robot == "pendubot":
    Q = np.diag([0.64, 0.56, 0.13, 0.037])
    R = np.eye(1)*0.82
Qf = np.copy(Q)

## PID controller
Kp = 10.
Ki = 0.
Kd = 0.1

## lqr controller
if robot == "acrobot":
    Q_lqr = np.diag((0.97, 0.93, 0.39, 0.26))
    R_lqr = np.diag((1.1, 1.1))
elif robot == "pendubot":
    Q_lqr = np.diag([0.0125, 6.5, 6.88, 9.36])
    R_lqr = np.diag([2.4, 2.4])



## ilqr mpc controller
N = 100
con_dt = dt
N_init = 100
max_iter = 5
max_iter_init = 1000
regu_init = 1.
max_regu = 10000.
min_regu = 0.01
break_cost_redu = 1e-6
trajectory_stabilization = False
shifting = 1
sCu = [0.0001, 0.0001]
sCp = [.1, .1]
sCv = [.01, .01]
sCen = 0.0
fCp = [10., 10.]
fCv = [1., 1.]
fCen = 0.0

def condition1(t, x):
    return False

def condition2(t, x):
    goal = [np.pi, 0., 0., 0.]
    #eps = [0.2, 0.2, 0.8, 0.8]
    eps = [0., 0., 0., 0.]

    y = wrap_angles_top(x)

    delta = np.abs(np.subtract(y, goal))
    max_diff = np.max(np.subtract(delta, eps))
    if max_diff > 0.:
        return False
    else:
        return True

timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", design, model, robot, "tvlqr_drake_and stabi", timestamp)
os.makedirs(save_dir)

# init plant, simulator and controller
plant = SymbolicDoublePendulum(model_pars=mpar)

sim = Simulator(plant=plant)
sim.set_process_noise(process_noise_sigmas=process_noise_sigmas)
sim.set_measurement_parameters(meas_noise_sigmas=meas_noise_sigmas,
                               delay=delay,
                               delay_mode=delay_mode)
sim.set_motor_parameters(u_noise_sigmas=u_noise_sigmas,
                         u_responsiveness=u_responsiveness)

controller1 = TVLQRController(
        csv_path=csv_path,
        urdf_path=urdf_path,
        model_pars=mpar_con,
        torque_limit=torque_limit,
        robot=robot,
        save_dir=save_dir)
controller1.set_cost_parameters(Q=Q, R=R, Qf=Qf)

if stabilization == "pid":
    controller2 = PointPIDController(
            torque_limit=torque_limit_pid,
            dt=dt)
    controller2.set_parameters(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd)
    controller2.set_goal(goal)
elif stabilization == "lqr":
    controller2 = LQRController(model_pars=mpar_con)
    controller2.set_goal(goal)
    controller2.set_cost_matrices(Q=Q_lqr, R=R_lqr)
    controller2.set_parameters(failure_value=0.0,
                              cost_to_go_cut=10000)

elif stabilization == "ilqr":
    controller2 = ILQRMPCCPPController(model_pars=mpar_con)
    controller2.set_goal(goal)
    controller2.set_parameters(N=N,
                               dt=con_dt,
                               max_iter=max_iter,
                               regu_init=regu_init,
                               max_regu=max_regu,
                               min_regu=min_regu,
                               break_cost_redu=break_cost_redu,
                               integrator=integrator,
                               trajectory_stabilization=trajectory_stabilization,
                               shifting=shifting)
    controller2.set_cost_parameters(sCu=sCu,
                                    sCp=sCp,
                                    sCv=sCv,
                                    sCen=sCen,
                                    fCp=fCp,
                                    fCv=fCv,
                                    fCen=fCen)

controller = CombinedController(
        controller1=controller1,
        controller2=controller2,
        condition1=condition1,
        condition2=condition2)
controller.set_filter_args(filt=meas_noise_vfilter, x0=goal, dt=dt, plant=plant,
                           simulator=sim, velocity_cut=meas_noise_cut,
                           filter_kwargs=filter_kwargs)
if friction_compensation:
    controller.set_friction_compensation(damping=mpar.b, coulomb_fric=mpar.cf)
    #controller.set_friction_compensation(damping=[0., 0.], coulomb_fric=mpar.cf)
controller.init()

# simulate
T, X, U = sim.simulate_and_animate(t0=0.0, x0=x0,
                                   tf=t_final, dt=dt, controller=controller,
                                   integrator="runge_kutta",
                                   plot_inittraj=True)

# saving and plotting
os.system(f"cp {csv_path} " + os.path.join(save_dir, "init_trajectory.csv"))
save_trajectory(os.path.join(save_dir, "trajectory.csv"), T, X, U)

plot_timeseries(T, X, U, None,
                plot_energy=False,
                T_des=T_des,
                X_des=X_des,
                U_des=U_des,
                X_filt=controller.x_filt_hist,
                X_meas=sim.meas_x_values,
                U_con=controller.u_hist,
                U_friccomp=controller.u_fric_hist,
                pos_y_lines=[0.0, np.pi],
                tau_y_lines=[-torque_limit[active_act], torque_limit[active_act]],
                save_to=os.path.join(save_dir, "timeseries"))
