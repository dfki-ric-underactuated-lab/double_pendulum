import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


#from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.utils.csv_trajectory import save_trajectory, load_trajectory, load_trajectory_full

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
                    X_filt=None,
                    U_con=None,
                    U_friccomp=None,
                    ACC_des=None,
                    save_to=None,
                    show=True
                    ):

    n_subplots = np.sum([plot_pos, plot_vel, plot_tau, plot_acc, plot_energy])

    SMALL_SIZE = 26
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 32

    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
        #"font.size": 26
    })

    fig, ax = plt.subplots(n_subplots,
                           1,
                           figsize=(16, n_subplots*3),
                           sharex="all")


    i = 0
    if plot_pos:
        #ax[i].plot(T[:len(X)], np.asarray(X).T[0], label=r"$q_1$", color="blue")
        #ax[i].plot(T[:len(X)], np.asarray(X).T[1], label=r"$q_2$", color="red")
        ax[i].plot(T[:len(X)], np.asarray(X).T[0], label="Joint 1", color="blue")
        ax[i].plot(T[:len(X)], np.asarray(X).T[1], label="Joint 2", color="red")
        if not (X_des is None):
            ax[i].plot(T_des[:len(X_des)], np.asarray(X_des).T[0],
                       ls="--", label=r"Joint 1 desired", color="lightblue")
                       #ls="--", label=r"$q_1$ desired", color="lightblue")
            ax[i].plot(T_des[:len(X_des)], np.asarray(X_des).T[1],
                       ls="--", label=r"Joint 2 desired", color="orange")
                       #ls="--", label=r"$q_2$ desired", color="orange")
        if not (X_meas is None):
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[0],
                       ls="-", label=r"$q_1$ measured", color="blue", alpha=0.2)
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[1],
                       ls="-", label=r"$q_2$ measured", color="red", alpha=0.2)
        if not (X_filt is None):
            ax[i].plot(T[:len(X_filt)], np.asarray(X_filt).T[0],
                       ls="-", label=r"$q_1$ filtered", color="darkblue")
            ax[i].plot(T[:len(X_filt)], np.asarray(X_filt).T[1],
                       ls="-", label=r"$q_2$ filtered", color="brown")
        for line in pos_x_lines:
            ax[i].plot([line, line], [np.min(X.T[:2]), np.max(X.T[:2])],
                       ls="--", color="gray")
        for line in pos_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("angle [rad]", fontsize=MEDIUM_SIZE)
        #ax[i].legend(loc="best")
        #ax[i].legend(bbox_to_anchor=(1.01,1), loc="upper left")
        ax[i].legend(bbox_to_anchor=(-0.05,1.5), loc="upper left", ncol=4)
    if plot_vel:
        i += 1
        ax[i].plot(T[:len(X)], np.asarray(X).T[2], label=r"$\dot{q}_1$", color="blue")
        ax[i].plot(T[:len(X)], np.asarray(X).T[3], label=r"$\dot{q}_1$", color="red")
        if not (X_des is None):
            ax[i].plot(T_des[:len(X_des)], np.asarray(X_des).T[2],
                       ls="--", label=r"$\dot{q}_1$ desired", color="lightblue")
            ax[i].plot(T_des[:len(X_des)], np.asarray(X_des).T[3],
                       ls="--", label=r"$\dot{q}_2$ desired", color="orange")
        if not (X_meas is None):
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[2],
                       ls="-", label=r"$\dot{q}_1$ measured", color="blue", alpha=0.2)
            ax[i].plot(T[:len(X_meas)], np.asarray(X_meas).T[3],
                       ls="-", label=r"$\dot{q}_2$ measured", color="red", alpha=0.2)
        if not (X_filt is None):
            ax[i].plot(T[:len(X_filt)], np.asarray(X_filt).T[2],
                       ls="-", label=r"$\dot{q}_1$ filtered", color="darkblue")
            ax[i].plot(T[:len(X_filt)], np.asarray(X_filt).T[3],
                       ls="-", label=r"$\dot{q}_2$ filtered", color="brown")
        for line in vel_x_lines:
            ax[i].plot([line, line], [np.min(X.T[2:]), np.max(X.T[2:])],
                       ls="--", color="gray")
        for line in vel_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("velocity [rad/s]", fontsize=MEDIUM_SIZE)
        #ax[i].legend(loc="best")
        #ax[i].legend(bbox_to_anchor=(1.01,1), loc="upper left")
    if plot_acc:
        i += 1
        ax[i].plot(T[:len(ACC)], np.asarray(ACC).T[0], label=r"$\ddot{q}_1$", color="blue")
        ax[i].plot(T[:len(ACC)], np.asarray(ACC).T[1], label=r"$\ddot{q}_2$", color="red")
        if not (ACC_des is None):
            ax[i].plot(T_des[:len(ACC_des)], np.asarray(ACC_des).T[0],
                       ls="--", label=r"$\ddot{q}_1$ desired", color="lightblue")
            ax[i].plot(T_des[:len(ACC_des)], np.asarray(ACC_des).T[1],
                       ls="--", label=r"$\ddot{q}_2$ desired", color="orange")
        for line in acc_x_lines:
            ax[i].plot([line, line], [np.min(X.T[2:]), np.max(X.T[2:])],
                       ls="--", color="gray")
        for line in acc_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("acceleration [rad/s^2]", fontsize=MEDIUM_SIZE)
        #ax[i].legend(loc="best")
        #ax[i].legend(bbox_to_anchor=(1.01,1), loc="upper left")
    if plot_tau:
        i += 1
        ax[i].plot(T[:len(U)], np.asarray(U).T[0, :len(T)], label=r"$u_1$", color="blue")
        ax[i].plot(T[:len(U)], np.asarray(U).T[1, :len(T)], label=r"$u_2$", color="red")
        if not (U_des is None):
            ax[i].plot(T_des[:len(U_des)], np.asarray(U_des).T[0],
                       ls="--", label=r"$u_1$ desired", color="lightblue")
            ax[i].plot(T_des[:len(U_des)], np.asarray(U_des).T[1],
                       ls="--", label=r"$u_2$ desired", color="orange")
        if not (U_con is None):
            ax[i].plot(T[:len(U_con)], np.asarray(U_con).T[0],
                       ls="-", label=r"$u_1$ controller", color="blue", alpha=0.2)
            ax[i].plot(T[:len(U_con)], np.asarray(U_con).T[1],
                       ls="-", label=r"$u_2$ controller", color="red", alpha=0.2)
        if not (U_friccomp is None):
            ax[i].plot(T[:len(U_friccomp)], np.asarray(U_friccomp).T[0],
                       ls="-", label=r"$u_1$ friction comp.", color="darkblue")
            ax[i].plot(T[:len(U_friccomp)], np.asarray(U_friccomp).T[1],
                       ls="-", label=r"$u_2$ friction comp.", color="brown")
        for line in tau_x_lines:
            ax[i].plot([line, line], [np.min(U), np.max(U)],
                       ls="--", color="gray")
        for line in tau_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("torque [Nm]", fontsize=MEDIUM_SIZE)
        #ax[i].legend(loc="best")
        #ax[i].legend(bbox_to_anchor=(1.01, 1.05), loc="upper left")
    if plot_energy:
        i += 1
        ax[i].plot(T[:len(energy)], np.asarray(energy), label="energy")
        for line in energy_x_lines:
            ax[i].plot([line, line], [np.min(energy), np.max(energy)],
                       ls="--", color="gray")
        for line in energy_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line],
                       ls="--", color="gray")
        ax[i].set_ylabel("energy [J]", fontsize=MEDIUM_SIZE)
        #ax[i].legend(loc="best")
        #ax[i].legend(bbox_to_anchor=(1.01,1), loc="upper left")
    ax[i].set_xlabel("time [s]", fontsize=MEDIUM_SIZE)
    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()
    plt.close()



traj = load_trajectory_full("../../data/experiment_records/design_A.0/20220819/175617-pendubot_tvlqr_lqr_VIDEO/trajectory.csv")
end = 3000 # len(traj["X_meas"])
T_des, X_des, U_des = load_trajectory("../../data/experiment_records/design_A.0/20220819/175617-pendubot_tvlqr_lqr_VIDEO/init_trajectory.csv")

plot_timeseries(traj["T"][:end], traj["X_meas"][:end], traj["U_meas"][:end],
              pos_y_lines=[0.0, np.pi],
              tau_y_lines=[-5., 5.],
              T_des=T_des, X_des=X_des, U_des=U_des,
              #U_con=traj["U_con"][:end],
              U_friccomp=traj["U_fric"][:end],
              save_to="figures/fig8_real_pendubot_swingup.pdf")

