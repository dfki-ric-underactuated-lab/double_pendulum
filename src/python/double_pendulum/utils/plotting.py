import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_timeseries(T, X=None, U=None, ACC=None, energy=None,
                    plot_pos=True,
                    plot_vel=True,
                    plot_acc=False,
                    plot_tau=True,
                    plot_energy=False,
                    pos_x_lines=[],
                    pos_y_lines=[],
                    vel_x_lines=[],
                    vel_y_lines=[],
                    acc_x_lines=[],
                    acc_y_lines=[],
                    tau_x_lines=[],
                    tau_y_lines=[],
                    energy_x_lines=[],
                    energy_y_lines=[],
                    T_des=None,
                    X_des=None,
                    U_des=None,
                    X_meas=None,
                    U_con=None,
                    ACC_des=None,
                    save_to=None,
                    ):

    n_subplots = np.sum([plot_pos, plot_vel, plot_tau, plot_acc, plot_energy])

    fig, ax = plt.subplots(n_subplots,
                           1,
                           figsize=(18, n_subplots*3),
                           sharex="all")

    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    i = 0
    if plot_pos:
        ax[i].plot(T, np.asarray(X).T[0], label="q1", color="blue")
        ax[i].plot(T, np.asarray(X).T[1], label="q2", color="red")
        if not (X_des is None):
            ax[i].plot(T_des, np.asarray(X_des).T[0],
                    ls="--", label="q1 desired", color="lightblue")
            ax[i].plot(T_des, np.asarray(X_des).T[1],
                    ls="--", label="q2 desired", color="orange")
        if not (X_meas is None):
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[0],
                    ls="-", label="q1 measured", color="blue", alpha=0.2)
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[1],
                    ls="-", label="q2 measured", color="red", alpha=0.2)
        for line in pos_x_lines:
            ax[i].plot([line, line], [np.min(X.T[:2]), np.max(X.T[:2])],
                       ls="--", color="gray")
        for line in pos_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("angle [rad]")
        ax[i].legend(loc="best")
    if plot_vel:
        i += 1
        ax[i].plot(T, np.asarray(X).T[2], label="q1 dot", color="blue")
        ax[i].plot(T, np.asarray(X).T[3], label="q2 dot", color="red")
        if not (X_des is None):
            ax[i].plot(T_des, np.asarray(X_des).T[2],
                    ls="--", label="q1 dot desired", color="lightblue")
            ax[i].plot(T_des, np.asarray(X_des).T[3],
                    ls="--", label="q2 dot desired", color="orange")
        if not (X_meas is None):
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[2],
                    ls="-", label="q1 dot measured", color="blue", alpha=0.2)
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[3],
                    ls="-", label="q2 dot measured", color="red", alpha=0.2)
        for line in vel_x_lines:
            ax[i].plot([line, line], [np.min(X.T[2:]), np.max(X.T[2:])],
                       ls="--", color="gray")
        for line in vel_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("angular velocity [rad/s]")
        ax[i].legend(loc="best")
    if plot_acc:
        i += 1
        ax[i].plot(T, np.asarray(ACC).T[0], label="q1 ddot", color="blue")
        ax[i].plot(T, np.asarray(ACC).T[1], label="q2 ddot", color="red")
        if not (ACC_des is None):
            ax[i].plot(T_des, np.asarray(ACC_des).T[0],
                    ls="--", label="q1 ddot desired", color="lightblue")
            ax[i].plot(T_des, np.asarray(ACC_des).T[1],
                    ls="--", label="q2 ddot desired", color="orange")
        for line in acc_x_lines:
            ax[i].plot([line, line], [np.min(X.T[2:]), np.max(X.T[2:])],
                       ls="--", color="gray")
        for line in acc_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("angular acceleration [rad/s^2]")
        ax[i].legend(loc="best")
    if plot_tau:
        i += 1
        ax[i].plot(T[:len(U)], np.asarray(U).T[0, :len(T)], label="u1", color="blue")
        ax[i].plot(T[:len(U)], np.asarray(U).T[1, :len(T)], label="u2", color="red")
        if not (U_des is None):
            ax[i].plot(T_des[:len(U_des)], np.asarray(U_des).T[0],
                    ls="--", label="u1 desired", color="lightblue")
            ax[i].plot(T_des[:len(U_des)], np.asarray(U_des).T[1],
                    ls="--", label="u2 desired", color="orange")
        if not (U_con is None):
            ax[i].plot(T[:len(U_con)], np.asarray(U_con).T[0],
                    ls="-", label="u1 controller", color="blue", alpha=0.2)
            ax[i].plot(T[:len(U_con)], np.asarray(U_con).T[1],
                    ls="-", label="u2 controller", color="red", alpha=0.2)
        for line in tau_x_lines:
            ax[i].plot([line, line], [np.min(U), np.max(U)],
                       ls="--", color="gray")
        for line in tau_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("input torque [Nm]")
        ax[i].legend(loc="best")
    if plot_energy:
        i += 1
        ax[i].plot(T, np.asarray(energy), label="energy")
        for line in energy_x_lines:
            ax[i].plot([line, line], [np.min(energy), np.max(energy)],
                       ls="--", color="gray")
        for line in energy_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("energy [J]")
        ax[i].legend(loc="best")
    ax[i].set_xlabel("time [s]")
    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    plt.show()


def plot_timeseries_csv(csv_path, read_with="pandas"):
    if read_with == "pandas":
        data = pd.read_csv(csv_path)
        time = data["time"].tolist()
        shoulder_pos = data["shoulder_pos"].tolist()
        shoulder_vel = data["shoulder_vel"].tolist()
        shoulder_trq = data["shoulder_torque"].tolist()

        elbow_pos = data["elbow_pos"].tolist()
        elbow_vel = data["elbow_vel"].tolist()
        elbow_trq = data["elbow_torque"].tolist()

        if "shoulder_acc" in data.keys() and "elbow_acc" in data.keys():
            shoulder_acc = data["shoulder_acc"].tolist()
            elbow_acc = data["elbow_acc"].tolist()
            ACC = np.asarray([shoulder_acc, elbow_acc]).T
            plot_acc = True
        else:
            plot_acc = False
            ACC = None

    X = np.asarray([shoulder_pos, elbow_pos, shoulder_vel, elbow_vel]).T
    U = np.asarray([shoulder_trq, elbow_trq]).T

    plot_timeseries(T=time,
                    X=X,
                    U=U,
                    ACC=ACC,
                    plot_acc=plot_acc)
