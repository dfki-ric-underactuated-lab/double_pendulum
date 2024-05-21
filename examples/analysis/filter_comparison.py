import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import load_trajectory

# from double_pendulum.utils.filters.low_pass import lowpass_filter_rt
# from double_pendulum.utils.filters.kalman_filter import kalman_filter_rt
# from double_pendulum.utils.filters.unscented_kalman_filter import (
#     unscented_kalman_filter_rt,
# )
from double_pendulum.filter.lowpass import lowpass_filter
from double_pendulum.filter.butterworth import butterworth_filter
from double_pendulum.filter.kalman import kalman_filter
from double_pendulum.filter.running_mean import runningmean_filter
from double_pendulum.filter.unscented_kalman import unscentedkalman_filter


csv_path = "../../data/trajectories/design_B.0/model_1.2/acrobot/noisy_trajectory.csv"
# T, X, U = load_trajectory(csv_path, "numpy")

data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
time_traj = data[:, 0]
pos1_traj = data[:, 1]
pos2_traj = data[:, 2]
vel1_traj = data[:, 3]
vel2_traj = data[:, 4]
tau1_traj = data[:, 5]
tau2_traj = data[:, 6]
meas_pos1_traj = data[:, 7]
meas_pos2_traj = data[:, 8]
meas_vel1_traj = data[:, 9]
meas_vel2_traj = data[:, 10]
T = time_traj.T
X = np.asarray([pos1_traj, pos2_traj, vel1_traj, vel2_traj]).T
X_meas = np.asarray([meas_pos1_traj, meas_pos2_traj, meas_vel1_traj, meas_vel2_traj]).T
U = np.asarray([tau1_traj, tau2_traj]).T


dt = T[1] - T[0]

cfric = [0.0, 0.0]
motor_inertia = 0.0
torque_limit = [0.0, 6.0]

model_par_path = "../../data/system_identification/identified_parameters/design_B.0/model_h2.0/model_parameters.yml"
mpar = model_parameters()
mpar.load_yaml(model_par_path)
mpar.set_motor_inertia(motor_inertia)
mpar.set_cfric(cfric)
mpar.set_torque_limit(torque_limit)

plant = SymbolicDoublePendulum(model_pars=mpar)
sim = Simulator(plant=plant)

process_noise = [0.0, 0.0, 0.0, 0.0]
measurement_noise = [0.1, 0.1, 0.2, 0.2]

lowpass = lowpass_filter(alpha=[1.0, 1.0, 0.3, 0.3], x0=X[0])
butterworth = butterworth_filter(cutoff=0.5, dt=dt, x0=X[0])
running_mean = runningmean_filter(window=10)
# kalman = kalman_filter_rt(
#         # plant=plant,
#         dim_x=4,
#         dim_u=2,
#         x0=X[0],
#         dt=dt,
#         process_noise=process_noise,
#         measurement_noise=measurement_noise)

unscented_kalman = unscentedkalman_filter(
    x0=X[0],
    dt=dt,
    process_noise=process_noise,
    measurement_noise=measurement_noise,
    fx=sim.runge_integrator,
)

X_lowpass = [X_meas[0]]
X_butterworth = [X_meas[0]]
X_running_mean = [X_meas[0]]
# X_kalman = [X_meas[0]]
X_ukalman = [X_meas[0]]
for i in range(1, len(X)):
    X_lowpass.append(lowpass.get_filtered_state(X_meas[i], U[i - 1]))
    X_butterworth.append(butterworth.get_filtered_state(X_meas[i], U[i - 1]))
    X_running_mean.append(running_mean.get_filtered_state(X_meas[i], U[i - 1]))
    # X_kalman.append(kalman(X_meas[i], U[i-1]))
    X_ukalman.append(unscented_kalman.get_filtered_state(X_meas[i], U[i - 1]))

X_lowpass = np.asarray(X_lowpass)
X_butterworth = np.asarray(X_butterworth)
X_running_mean = np.asarray(X_running_mean)
# X_kalman = np.asarray(X_kalman)
X_ukalman = np.asarray(X_ukalman)

fig, ax = plt.subplots(4, 1, figsize=(10, 6), sharex="all")

for i in range(4):
    ax[i].plot(T, X.T[i], label="true data", color="blue", alpha=1.0)
    ax[i].plot(T, X_meas.T[i], label="measured data", color="lightblue")
    ax[i].plot(T, X_lowpass.T[i], label="lowpass filter", color="red", lw=0.5)
    ax[i].plot(
        T, X_butterworth.T[i], label="butterworth filter", color="purple", lw=0.5
    )
    ax[i].plot(
        T, X_running_mean.T[i], label="running mean filter", color="orange", lw=0.5
    )
    # ax[i].plot(T, X_kalman.T[i], label="kalman filter", color="orange", lw=0.5)
    ax[i].plot(
        T, X_ukalman.T[i], label="unscented kalman filter", color="brown", lw=0.5
    )
ax[0].legend(bbox_to_anchor=(1.01, 1))
plt.tight_layout()

plt.show()
