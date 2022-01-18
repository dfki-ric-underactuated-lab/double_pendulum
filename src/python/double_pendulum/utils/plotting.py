import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_timeseries(T, X=None, U=None, energy=None,
                    plot_pos=True,
                    plot_vel=True,
                    plot_tau=True,
                    plot_energy=False,
                    pos_x_lines=[],
                    pos_y_lines=[],
                    vel_x_lines=[],
                    vel_y_lines=[],
                    tau_x_lines=[],
                    tau_y_lines=[],
                    energy_x_lines=[],
                    energy_y_lines=[],
                    ):

    n_subplots = np.sum([plot_pos, plot_vel, plot_tau, plot_energy])

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
        ax[i].plot(T, np.asarray(X).T[0], label="q1")
        ax[i].plot(T, np.asarray(X).T[1], label="q2")
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
        ax[i].plot(T, np.asarray(X).T[2], label="q1 dot")
        ax[i].plot(T, np.asarray(X).T[3], label="q2 dot")
        for line in vel_x_lines:
            ax[i].plot([line, line], [np.min(X.T[2:]), np.max(X.T[2:])],
                       ls="--", color="gray")
        for line in vel_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("angular velocity [rad/s]")
        ax[i].legend(loc="best")
    if plot_tau:
        i += 1
        ax[i].plot(T, np.asarray(U).T[0], label="u1")
        ax[i].plot(T, np.asarray(U).T[1], label="u2")
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
    plt.show()
