import numpy as np
import pandas as pd


def save_trajectory(csv_path, T, X, U):
    TT = np.asarray(T)
    XX = np.asarray(X)
    UU = np.asarray(U)
    data = np.asarray([TT,
                       XX.T[0], XX.T[1], XX.T[2], XX.T[3],
                       UU.T[0], UU.T[1]]).T
    np.savetxt(csv_path, data, delimiter=",",
               header="time, pos1, pos2, vel1, vel2, tau1, tau2")


def load_trajectory(csv_path, read_with="numpy",
                    with_tau=True, keys="shoulder-elbow"):
    if read_with == "pandas":
        data = pd.read_csv(csv_path)

        if keys == "shoulder-elbow":
            time_traj = np.asarray(data["time"])
            pos1_traj = np.asarray(data["shoulder_pos"])
            pos2_traj = np.asarray(data["elbow_pos"])
            vel1_traj = np.asarray(data["shoulder_vel"])
            vel2_traj = np.asarray(data["elbow_vel"])
            if with_tau:
                tau1_traj = np.asarray(data["shoulder_torque"])
                tau2_traj = np.asarray(data["elbow_torque"])
        else:
            time_traj = np.asarray(data["time"])
            pos1_traj = np.asarray(data["pos1"])
            pos2_traj = np.asarray(data["pos2"])
            vel1_traj = np.asarray(data["vel1"])
            vel2_traj = np.asarray(data["vel2"])
            if with_tau:
                tau1_traj = np.asarray(data["tau1"])[:-1]
                tau2_traj = np.asarray(data["tau2"])[:-1]

    elif read_with == "numpy":
        data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

        time_traj = data[:, 0]
        pos1_traj = data[:, 1]
        pos2_traj = data[:, 2]
        vel1_traj = data[:, 3]
        vel2_traj = data[:, 4]
        if with_tau:
            tau1_traj = data[:-1, 5]
            tau2_traj = data[:-1, 6]

    T = time_traj.T
    X = np.asarray([pos1_traj, pos2_traj,
                    vel1_traj, vel2_traj]).T
    if with_tau:
        U = np.asarray([tau1_traj, tau2_traj]).T
    else:
        U = np.asarray([np.zeros_like(T), np.zeros_like(T)])
    return T, X, U
