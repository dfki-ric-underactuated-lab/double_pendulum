import numpy as np
import pandas as pd


# def save_trajectory(csv_path, T, X, U=None):
#     TT = np.asarray(T)
#     XX = np.asarray(X)
#     if U is not None:
#         UU = np.asarray(U)
#         if len(UU) < len(XX):
#             UU = np.append(UU, [UU[-1]], axis=0)
#         data = np.asarray([TT,
#                            XX.T[0], XX.T[1], XX.T[2], XX.T[3],
#                            UU.T[0], UU.T[1]]).T
#         np.savetxt(csv_path, data, delimiter=",",
#                    header="time,pos1,pos2,vel1,vel2,tau1,tau2")
#     else:
#         data = np.asarray([TT,
#                            XX.T[0], XX.T[1], XX.T[2], XX.T[3]]).T
#         np.savetxt(csv_path, data, delimiter=",",
#                    header="time,pos1,pos2,vel1,vel2")

# def save_trajectory(csv_path, T, X, U=None):
#     save_trajectory_full(csv_path,
#                          time=T,
#                          X=X,
#                          U=U)


def save_trajectory(
    csv_path,
    T=None,
    X=None,
    U=None,
    ACC=None,
    X_meas=None,
    X_filt=None,
    X_des=None,
    U_con=None,
    U_fric=None,
    U_meas=None,
    U_des=None,
    U_perturbation=None,
    K=None,
    k=None,
):
    data = []
    header = ""

    min_len = np.inf

    if T is not None and len(T) > 0:
        data.append(np.array(T))
        header += "time"
        min_len = min(min_len, len(T))

    if X is not None and len(X) > 0:
        data.append(np.array(X).T[0])
        data.append(np.array(X).T[1])
        data.append(np.array(X).T[2])
        data.append(np.array(X).T[3])
        header += ",pos1,pos2,vel1,vel2"
        min_len = min(min_len, len(X))

    if U is not None and len(U) > 0:
        data.append(np.array(U).T[0])
        data.append(np.array(U).T[1])
        header += ",tau1,tau2"
        min_len = min(min_len, len(U))

    if ACC is not None and len(ACC) > 0:
        data.append(np.array(ACC).T[0])
        data.append(np.array(ACC).T[1])
        header += ",acc1,acc2"
        min_len = min(min_len, len(ACC))

    if X_meas is not None and len(X_meas) > 0:
        data.append(np.array(X_meas).T[0])
        data.append(np.array(X_meas).T[1])
        data.append(np.array(X_meas).T[2])
        data.append(np.array(X_meas).T[3])
        header += ",pos_meas1,pos_meas2,vel_meas1,vel_meas2"
        min_len = min(min_len, len(X_meas))

    if X_filt is not None and len(X_filt) > 0:
        data.append(np.array(X_filt).T[0])
        data.append(np.array(X_filt).T[1])
        data.append(np.array(X_filt).T[2])
        data.append(np.array(X_filt).T[3])
        header += ",pos_filt1,pos_filt2,vel_filt1,vel_filt2"
        min_len = min(min_len, len(X_filt))

    if X_des is not None and len(X_des) > 0:
        if len(X_des) < min_len:
            diff_len = min_len - len(X_des)
            X_des = np.append(X_des, np.zeros((diff_len, 4)), axis=0)
        data.append(np.array(X_des).T[0])
        data.append(np.array(X_des).T[1])
        data.append(np.array(X_des).T[2])
        data.append(np.array(X_des).T[3])
        header += ",pos_des1,pos_des2,vel_des1,vel_des2"
        min_len = min(min_len, len(X_des))

    if U_con is not None and len(U_con) > 0:
        data.append(np.array(U_con).T[0])
        data.append(np.array(U_con).T[1])
        header += ",tau_con1,tau_con2"
        min_len = min(min_len, len(U_con))

    if U_fric is not None and len(U_fric) > 0:
        data.append(np.array(U_fric).T[0])
        data.append(np.array(U_fric).T[1])
        header += ",tau_fric1,tau_fric2"
        min_len = min(min_len, len(U_fric))

    if U_meas is not None and len(U_meas) > 0:
        data.append(np.array(U_meas).T[0])
        data.append(np.array(U_meas).T[1])
        header += ",tau_meas1,tau_meas2"
        min_len = min(min_len, len(U_meas))

    if U_des is not None and len(U_des) > 0:
        if len(U_des) < min_len:
            diff_len = min_len - len(U_des)
            U_des = np.append(U_des, np.zeros((diff_len, 2)), axis=0)
        data.append(np.array(U_des).T[0])
        data.append(np.array(U_des).T[1])
        header += ",tau_des1,tau_des2"
        min_len = min(min_len, len(U_des))

    if U_perturbation is not None and len(U_perturbation) > 0:
        if len(U_perturbation) < min_len:
            diff_len = min_len - len(U_perturbation)
            U_perturbation = np.append(U_perturbation, np.zeros((diff_len, 2)), axis=0)
        data.append(np.array(U_perturbation).T[0])
        data.append(np.array(U_perturbation).T[1])
        header += ",tau_pert1,tau_pert2"
        min_len = min(min_len, len(U_perturbation))

    if K is not None and len(K) > 0:
        data.append(np.array(K).T[0, 0])
        data.append(np.array(K).T[0, 1])
        data.append(np.array(K).T[0, 2])
        data.append(np.array(K).T[0, 3])
        data.append(np.array(K).T[1, 0])
        data.append(np.array(K).T[1, 1])
        data.append(np.array(K).T[1, 2])
        data.append(np.array(K).T[1, 3])
        header += ",K11,K12,K13,K14,K21,K22,K23,K24"
        min_len = min(min_len, len(K))

    if k is not None and len(k) > 0:
        data.append(np.array(k).T[0])
        data.append(np.array(k).T[1])
        header += ",k1,k2"
        min_len = min(min_len, len(k))

    for i, d in enumerate(data):
        data[i] = data[i][:min_len]

    data = np.array(data).T

    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    # print(f"CSV file saved to {csv_path}")


def load_trajectory_full(csv_path):
    traj = {}
    # traj["time"] = None
    # traj["pos1"] = None
    # traj["pos2"] = None
    # traj["vel1"] = None
    # traj["vel2"] = None
    # traj["tau1"] = None
    # traj["tau2"] = None
    # traj["acc1"] = None
    # traj["acc2"] = None
    # traj["tau_con1"] = None
    # traj["tau_con2"] = None
    # traj["tau_fric1"] = None
    # traj["tau_fric2"] = None
    # traj["K1"] = None
    # traj["K2"] = None
    # traj["k1"] = None
    # traj["k2"] = None

    traj["T"] = None
    traj["X"] = None
    traj["U"] = None
    traj["ACC"] = None
    traj["X_meas"] = None
    traj["X_filt"] = None
    traj["X_des"] = None
    traj["U_con"] = None
    traj["U_fric"] = None
    traj["U_meas"] = None
    traj["U_des"] = None
    traj["K"] = None
    traj["k"] = None

    data = pd.read_csv(csv_path)

    if "time" in data.keys():
        traj["T"] = np.asarray(data["time"])

    if all((key in data.keys() for key in ["pos1", "pos2", "vel1", "vel2"])):
        traj["X"] = np.asarray(
            [data["pos1"], data["pos2"], data["vel1"], data["vel2"]]
        ).T

    if all((key in data.keys() for key in ["tau1", "tau2"])):
        traj["U"] = np.asarray([data["tau1"], data["tau2"]]).T

    if all((key in data.keys() for key in ["acc1", "acc2"])):
        traj["ACC"] = np.asarray([data["acc1"], data["acc2"]]).T

    if all(
        (
            key in data.keys()
            for key in ["pos_meas1", "pos_meas2", "vel_meas1", "vel_meas2"]
        )
    ):
        traj["X_meas"] = np.asarray(
            [data["pos_meas1"], data["pos_meas2"], data["vel_meas1"], data["vel_meas2"]]
        ).T

    if all(
        (
            key in data.keys()
            for key in ["pos_filt1", "pos_filt2", "vel_filt1", "vel_filt2"]
        )
    ):
        traj["X_filt"] = np.asarray(
            [data["pos_filt1"], data["pos_filt2"], data["vel_filt1"], data["vel_filt2"]]
        ).T

    if all(
        (key in data.keys() for key in ["pos_des1", "pos_des2", "vel_des1", "vel_des2"])
    ):
        traj["X_des"] = np.asarray(
            [data["pos_des1"], data["pos_des2"], data["vel_des1"], data["vel_des2"]]
        ).T

    if all((key in data.keys() for key in ["tau_con1", "tau_con2"])):
        traj["U_con"] = np.asarray([data["tau_con1"], data["tau_con2"]]).T

    if all((key in data.keys() for key in ["tau_fric1", "tau_fric2"])):
        traj["U_fric"] = np.asarray([data["tau_fric1"], data["tau_fric2"]]).T

    if all((key in data.keys() for key in ["tau_meas1", "tau_meas2"])):
        traj["U_meas"] = np.asarray([data["tau_meas1"], data["tau_meas2"]]).T

    if all((key in data.keys() for key in ["tau_des1", "tau_des2"])):
        traj["U_des"] = np.asarray([data["tau_des1"], data["tau_des2"]]).T

    if all(
        (
            key in data.keys()
            for key in ["K11", "K12", "K13", "K14", "K21", "K22", "K23", "K24"]
        )
    ):
        K11 = np.asarray(data["K11"])
        K12 = np.asarray(data["K12"])
        K13 = np.asarray(data["K13"])
        K14 = np.asarray(data["K14"])
        K21 = np.asarray(data["K21"])
        K22 = np.asarray(data["K22"])
        K23 = np.asarray(data["K23"])
        K24 = np.asarray(data["K24"])
        K = np.asarray([[K11, K12, K13, K14], [K21, K22, K23, K24]])
        traj["K"] = np.swapaxes(K, 0, 2)

    if all((key in data.keys() for key in ["k1", "k2"])):
        traj["k"] = np.asarray([data["k1"], data["k2"]]).T

    return traj


def load_trajectory(csv_path, with_tau=True):
    traj = load_trajectory_full(csv_path)

    T = traj["T"]
    X = traj["X"]
    if with_tau:
        U = traj["U"]
    else:
        U = np.asarray([np.zeros_like(T), np.zeros_like(T)]).T
    return T, X, U


# def load_trajectory(csv_path, read_with="numpy",
#                     with_tau=True, keys="shoulder-elbow",
#                     resample=False, resample_dt=None,
#                     num_break=40, poly_degree=3):
#     if read_with == "pandas":
#         data = pd.read_csv(csv_path)
#
#         if keys == "shoulder-elbow":
#             time_traj = np.asarray(data["time"])
#             pos1_traj = np.asarray(data["shoulder_pos"])
#             pos2_traj = np.asarray(data["elbow_pos"])
#             vel1_traj = np.asarray(data["shoulder_vel"])
#             vel2_traj = np.asarray(data["elbow_vel"])
#             if with_tau:
#                 tau1_traj = np.asarray(data["shoulder_torque"])
#                 tau2_traj = np.asarray(data["elbow_torque"])
#         else:
#             time_traj = np.asarray(data["time"])
#             pos1_traj = np.asarray(data["pos1"])
#             pos2_traj = np.asarray(data["pos2"])
#             vel1_traj = np.asarray(data["vel1"])
#             vel2_traj = np.asarray(data["vel2"])
#             if with_tau:
#                 tau1_traj = np.asarray(data["tau1"])
#                 tau2_traj = np.asarray(data["tau2"])
#
#     elif read_with == "numpy":
#         data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
#
#         time_traj = data[:, 0]
#         pos1_traj = data[:, 1]
#         pos2_traj = data[:, 2]
#         vel1_traj = data[:, 3]
#         vel2_traj = data[:, 4]
#         if with_tau:
#             tau1_traj = data[:, 5]
#             tau2_traj = data[:, 6]
#
#     T = time_traj.T
#     X = np.asarray([pos1_traj, pos2_traj,
#                     vel1_traj, vel2_traj]).T
#     if with_tau:
#         U = np.asarray([tau1_traj, tau2_traj]).T
#     else:
#         U = np.asarray([np.zeros_like(T), np.zeros_like(T)])
#
#     if resample:
#         T, X, U = ResampleTrajectory(T, X, U, resample_dt, num_break, poly_degree)
#     return T, X, U
#
#
# def load_acceleration(csv_path, read_with="numpy",
#                       keys="shoulder-elbow"):
#     if read_with == "pandas":
#         data = pd.read_csv(csv_path)
#
#         if "shoulder_acc" in data.keys() and "elbow_acc" in data.keys():
#             shoulder_acc = data["shoulder_acc"].tolist()
#             elbow_acc = data["elbow_acc"].tolist()
#             ACC = np.asarray([shoulder_acc, elbow_acc]).T
#         else:
#             ACC = None
#     elif read_with == "numpy":
#         data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
#         if np.shape(data)[1] >= 8:
#             shoulder_acc = data[:, 7]
#             elbow_acc = data[:, 8]
#             ACC = np.asarray([shoulder_acc, elbow_acc]).T
#         else:
#             ACC = None
#
#     return ACC


def concatenate_trajectories(csv_paths=[], with_tau=True):
    if type(csv_paths) == str:
        # csv_paths = [csv_paths]
        T_out, X_out, U_out = load_trajectory(csv_path=csv_path, with_tau=with_tau)
    elif type(csv_paths) == list:
        T_outs = []
        X_outs = []
        U_outs = []
        time = 0.0
        for i, cp in enumerate(csv_paths):
            traj = load_trajectory_full(cp)

            T = traj["T"]

            if traj["X"] is not None:
                X = traj["X"]
            else:
                X = traj["X_meas"]

            if traj["U"] is not None:
                U = traj["U"]
            else:
                U = traj["U_meas"]

            T_outs.append(T + time)
            time = T_outs[-1][-1] + (T_outs[-1][-1] - T_outs[-1][-2])
            X_outs.append(X)
            U_outs.append(U)

        T_out = np.concatenate(T_outs, axis=0)
        X_out = np.concatenate(X_outs, axis=0)
        U_out = np.concatenate(U_outs, axis=0)

    else:
        T_out = None
        X_out = None
        U_out = None

    return T_out, X_out, U_out


def trajectory_properties(T, X):
    dt = T[1] - T[0]
    t_final = T[-1]

    x0 = X[0]
    xf = X[-1]
    return dt, t_final, x0, xf


def load_Kk_values(csv_path):
    traj = load_trajectory_full(csv_path)
    K1 = traj["K"][:, :, 0]
    K2 = traj["K"][:, :, 1]

    k1 = traj["k"][:, 0]
    k2 = traj["k"][:, 0]
    return K1, K2, k1, k2


# def load_Kk_values(csv_path, read_with, keys=""):
#     if read_with == "pandas":
#         print("loading of kK values with pandas not yet implemented")
#     elif read_with == "numpy":
#         data = np.loadtxt(csv_path, skiprows=1, delimiter=",")
#
#         K11 = data[:, 7]
#         K12 = data[:, 8]
#         K13 = data[:, 9]
#         K14 = data[:, 10]
#         K21 = data[:, 11]
#         K22 = data[:, 12]
#         K23 = data[:, 13]
#         K24 = data[:, 14]
#         K1 = np.asarray([K11, K12, K13, K14])
#         K2 = np.asarray([K21, K22, K23, K24])
#         K1 = np.swapaxes(K1, 0, 1)
#         K2 = np.swapaxes(K2, 0, 1)
#
#         k1 = data[:, 15]
#         k2 = data[:, 16]
#
#     return K1, K2, k1, k2
