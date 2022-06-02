import numpy as np
import pandas as pd


def save_trajectory(csv_path, T, X, U):
    TT = np.asarray(T)
    XX = np.asarray(X)
    UU = np.asarray(U)
    if len(UU) < len(XX):
        UU = np.append(UU, [UU[-1]], axis=0)
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
                tau1_traj = np.asarray(data["tau1"])
                tau2_traj = np.asarray(data["tau2"])

    elif read_with == "numpy":
        data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

        time_traj = data[:, 0]
        pos1_traj = data[:, 1]
        pos2_traj = data[:, 2]
        vel1_traj = data[:, 3]
        vel2_traj = data[:, 4]
        if with_tau:
            tau1_traj = data[:, 5]
            tau2_traj = data[:, 6]

    T = time_traj.T
    X = np.asarray([pos1_traj, pos2_traj,
                    vel1_traj, vel2_traj]).T
    if with_tau:
        U = np.asarray([tau1_traj, tau2_traj]).T
    else:
        U = np.asarray([np.zeros_like(T), np.zeros_like(T)])
    return T, X, U

def trajectory_properties(T, X):

    dt = T[1] - T[0]
    t_final = T[-1]

    x0 = X[0]
    xf = X[-1]
    return dt, t_final, x0, xf


def load_Kk_values(csv_path, read_with, keys=""):
    if read_with == "pandas":
        print("loading of kK values with pandas not yet implemented")
    elif read_with == "numpy":
        data = np.loadtxt(csv_path, skiprows=1, delimiter=",")

        K11 = data[:, 7]
        K12 = data[:, 8]
        K13 = data[:, 9]
        K14 = data[:, 10]
        K21 = data[:, 11]
        K22 = data[:, 12]
        K23 = data[:, 13]
        K24 = data[:, 14]
        K1 = np.asarray([K11, K12, K13, K14])
        K2 = np.asarray([K21, K22, K23, K24])
        K1 = np.swapaxes(K1, 0, 1)
        K2 = np.swapaxes(K2, 0, 1)

        k1 = data[:, 15]
        k2 = data[:, 16]

    return K1, K2, k1, k2
