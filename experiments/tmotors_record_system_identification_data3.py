import numpy as np

from double_pendulum.controller.pid.trajectory_pid_controller import TrajPIDController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment

design = "design_A.0"

torque_limit = [5.0, 5.0]

# trajectory
dt = 0.002
t_final = 20.0
t1_final = 5.0
N = int(t1_final / dt)
T_des = np.linspace(0, t1_final, N+1)
p1_des = np.linspace(0, -np.pi/2, N+1)
p2_des = np.linspace(0, -np.pi/2, N+1)
v1_des = np.diff(p1_des, append=p1_des[-1]) / dt
v2_des = np.diff(p2_des, append=p2_des[-1]) / dt
X_des = np.array([p1_des, p2_des, v1_des, v2_des]).T

# controller parameters
Kp = 20.
Ki = 0.
Kd = 1.


def condition1(t, x):
    return False


def condition2(t, x):
    return t > 5.0


# controller
controller1 = TrajPIDController(T=T_des,
                                X=X_des,
                                use_feed_forward_torque=False,
                                torque_limit=torque_limit,
                                num_break=40)
controller1.set_parameters(Kp=Kp, Ki=Ki, Kd=Kd)

controller2 = TrajPIDController(T=T_des,
                                X=X_des,
                                use_feed_forward_torque=False,
                                torque_limit=torque_limit,
                                num_break=40)
controller2.set_parameters(Kp=0., Ki=0., Kd=0.)

controller = CombinedController(
        controller1=controller1,
        controller2=controller2,
        condition1=condition1,
        condition2=condition2)
controller.init()

# experiment
run_experiment(controller=controller,
               dt=dt,
               t_final=t_final,
               can_port="can0",
               motor_ids=[7, 8],
               tau_limit=torque_limit,
               save_dir="data/"+design+"/double-pendulum/tmotors/sysid")
