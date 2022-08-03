import numpy as np
import pandas as pd

from double_pendulum.utils.pcw_polynomial import ResampleTrajectory

def save_trajectory(csv_path, T, X, U=None):
    TT = np.asarray(T)
    XX = np.asarray(X)
    if U is not None:
        UU = np.asarray(U)
        if len(UU) < len(XX):
            UU = np.append(UU, [UU[-1]], axis=0)
        data = np.asarray([TT,
                           XX.T[0], XX.T[1], XX.T[2], XX.T[3],
                           UU.T[0], UU.T[1]]).T
        np.savetxt(csv_path, data, delimiter=",",
                   header="time, pos1, pos2, vel1, vel2, tau1, tau2")
    else:
        data = np.asarray([TT,
                           XX.T[0], XX.T[1], XX.T[2], XX.T[3]]).T
        np.savetxt(csv_path, data, delimiter=",",
                   header="time, pos1, pos2, vel1, vel2")


def load_trajectory(csv_path, read_with="numpy",
                    with_tau=True, keys="shoulder-elbow",
                    resample=False, resample_dt=None,
                    num_break=40, poly_degree=3):
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

    if resample:
        T, X, U = ResampleTrajectory(T, X, U, resample_dt, num_break, poly_degree)
    return T, X, U


def load_acceleration(csv_path, read_with="numpy",
                      keys="shoulder-elbow"):
    if read_with == "pandas":
        data = pd.read_csv(csv_path)

        if "shoulder_acc" in data.keys() and "elbow_acc" in data.keys():
            shoulder_acc = data["shoulder_acc"].tolist()
            elbow_acc = data["elbow_acc"].tolist()
            ACC = np.asarray([shoulder_acc, elbow_acc]).T
        else:
            ACC = None
    elif read_with == "numpy":
        data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
        if np.shape(data)[1] >= 8:
            shoulder_acc = data[:, 7]
            elbow_acc = data[:, 8]
            ACC = np.asarray([shoulder_acc, elbow_acc]).T
        else:
            ACC = None

    return ACC


def concatenate_trajectories(csv_paths=[], read_withs="numpy",
                             with_tau=True, keys="",
                             save_to="conc_trajectrory.csv"):
    if type(csv_paths) != list:
        csv_paths = [csv_paths]
    n = len(csv_paths)

    if type(read_withs) != list:
        read_withs = n*[read_withs]

    if type(keys) != list:
        keys = n*[keys]

    T_outs = []
    X_outs = []
    U_outs = []
    time = 0.
    for i, cp in enumerate(csv_paths):
        t, x, u = load_trajectory(csv_path=cp,
                                  read_with=read_withs[i],
                                  with_tau=with_tau,
                                  keys=keys[i])
        T_outs.append(t + time)
        time = t[-1] + (t[-1] - t[-2])
        X_outs.append(x)
        U_outs.append(u)

    T_out = np.concatenate(T_outs, axis=0)
    X_out = np.concatenate(X_outs, axis=0)
    U_out = np.concatenate(U_outs, axis=0)

    return T_out, X_out, U_out


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
