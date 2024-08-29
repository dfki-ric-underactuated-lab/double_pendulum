import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_timeseries(
    T,
    X=None,
    U=None,
    ACC=None,
    energy=None,
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
    U_perturbation=None,
    ACC_des=None,
    save_to=None,
    show=True,
    scale=1.0,
):
    n_subplots = np.sum([plot_pos, plot_vel, plot_tau, plot_acc, plot_energy])

    SMALL_SIZE = 16 * scale
    MEDIUM_SIZE = 20 * scale
    BIGGER_SIZE = 24 * scale

    mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
    mpl.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    mpl.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(
        n_subplots, 1, figsize=(18 * scale, n_subplots * 3 * scale), sharex="all"
    )

    i = 0
    if plot_pos:
        ax[i].plot(T[: len(X)], np.asarray(X).T[0], label=r"$q_1$", color="blue")
        ax[i].plot(T[: len(X)], np.asarray(X).T[1], label=r"$q_2$", color="red")
        if not (X_des is None or np.shape(X_des)[0] == 0):
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[0],
                ls="--",
                label=r"$q_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[1],
                ls="--",
                label=r"$q_2$ desired",
                color="orange",
            )
        if not (X_meas is None or np.shape(X_meas)[0] == 0):
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[0],
                ls="-",
                label=r"$q_1$ measured",
                color="blue",
                alpha=0.2,
            )
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[1],
                ls="-",
                label=r"$q_2$ measured",
                color="red",
                alpha=0.2,
            )
        if not (X_filt is None or np.shape(X_filt)[0] == 0):
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[0],
                ls="-",
                label=r"$q_1$ filtered",
                color="darkblue",
            )
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[1],
                ls="-",
                label=r"$q_2$ filtered",
                color="brown",
            )
        for line in pos_x_lines:
            ax[i].plot(
                [line, line], [np.min(X.T[:2]), np.max(X.T[:2])], ls="--", color="gray"
            )
        for line in pos_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("angle [rad]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_vel:
        i += 1
        ax[i].plot(T[: len(X)], np.asarray(X).T[2], label=r"$\dot{q}_1$", color="blue")
        ax[i].plot(T[: len(X)], np.asarray(X).T[3], label=r"$\dot{q}_2$", color="red")
        if not (X_des is None or np.shape(X_des)[0] == 0):
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[2],
                ls="--",
                label=r"$\dot{q}_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[3],
                ls="--",
                label=r"$\dot{q}_2$ desired",
                color="orange",
            )
        if not (X_meas is None or np.shape(X_meas)[0] == 0):
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[2],
                ls="-",
                label=r"$\dot{q}_1$ measured",
                color="blue",
                alpha=0.2,
            )
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[3],
                ls="-",
                label=r"$\dot{q}_2$ measured",
                color="red",
                alpha=0.2,
            )
        if not (X_filt is None or np.shape(X_filt)[0] == 0):
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[2],
                ls="-",
                label=r"$\dot{q}_1$ filtered",
                color="darkblue",
            )
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[3],
                ls="-",
                label=r"$\dot{q}_2$ filtered",
                color="brown",
            )
        for line in vel_x_lines:
            ax[i].plot(
                [line, line], [np.min(X.T[2:]), np.max(X.T[2:])], ls="--", color="gray"
            )
        for line in vel_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("velocity [rad/s]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_acc:
        i += 1
        ax[i].plot(
            T[: len(ACC)], np.asarray(ACC).T[0], label=r"$\ddot{q}_1$", color="blue"
        )
        ax[i].plot(
            T[: len(ACC)], np.asarray(ACC).T[1], label=r"$\ddot{q}_2$", color="red"
        )
        if not (ACC_des is None or np.shape(ACC_des)[0] == 0):
            ax[i].plot(
                T_des[: len(ACC_des)],
                np.asarray(ACC_des).T[0],
                ls="--",
                label=r"$\ddot{q}_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(ACC_des)],
                np.asarray(ACC_des).T[1],
                ls="--",
                label=r"$\ddot{q}_2$ desired",
                color="orange",
            )
        for line in acc_x_lines:
            ax[i].plot(
                [line, line], [np.min(X.T[2:]), np.max(X.T[2:])], ls="--", color="gray"
            )
        for line in acc_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("acceleration [rad/s^2]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_tau:
        i += 1
        ax[i].plot(
            T[: len(U)], np.asarray(U).T[0, : len(T)], label=r"$u_1$", color="blue"
        )
        ax[i].plot(
            T[: len(U)], np.asarray(U).T[1, : len(T)], label=r"$u_2$", color="red"
        )
        if not (U_des is None or np.shape(U_des)[0] == 0):
            ax[i].plot(
                T_des[: len(U_des)],
                np.asarray(U_des).T[0],
                ls="--",
                label=r"$u_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(U_des)],
                np.asarray(U_des).T[1],
                ls="--",
                label=r"$u_2$ desired",
                color="lightcoral",
            )
        if not (U_con is None or np.shape(U_con)[0] == 0):
            ax[i].plot(
                T[: len(U_con)],
                np.asarray(U_con).T[0],
                ls="-",
                label=r"$u_1$ controller",
                color="blue",
                alpha=0.2,
            )
            ax[i].plot(
                T[: len(U_con)],
                np.asarray(U_con).T[1],
                ls="-",
                label=r"$u_2$ controller",
                color="red",
                alpha=0.2,
            )
        if not (U_friccomp is None or np.shape(U_friccomp)[0] == 0):
            ax[i].plot(
                T[: len(U_friccomp)],
                np.asarray(U_friccomp).T[0],
                ls="-",
                label=r"$u_1$ friction comp.",
                color="darkblue",
            )
            ax[i].plot(
                T[: len(U_friccomp)],
                np.asarray(U_friccomp).T[1],
                ls="-",
                label=r"$u_2$ friction comp.",
                color="brown",
            )
        if not (U_perturbation is None or np.shape(U_perturbation)[0] == 0):
            ax[i].plot(
                T,
                np.asarray(U_perturbation).T[0][: len(T)],
                ls="-",
                label=r"$u_1$ perturbation",
                color="green",
            )
            ax[i].plot(
                T,
                np.asarray(U_perturbation).T[1][: len(T)],
                ls="-",
                label=r"$u_2$ perturbation",
                color="orange",
            )
        for line in tau_x_lines:
            ax[i].plot([line, line], [np.min(U), np.max(U)], ls="--", color="gray")
        for line in tau_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("torque [Nm]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_energy:
        i += 1
        ax[i].plot(T[: len(energy)], np.asarray(energy), label="energy")
        for line in energy_x_lines:
            ax[i].plot(
                [line, line], [np.min(energy), np.max(energy)], ls="--", color="gray"
            )
        for line in energy_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("energy [J]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax[i].set_xlabel("time [s]", fontsize=MEDIUM_SIZE)
    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()
    plt.close()


def plot_figures(
    save_dir,
    index,
    meas_time,
    shoulder_meas_pos,
    shoulder_meas_vel,
    shoulder_meas_tau,
    elbow_meas_pos,
    elbow_meas_vel,
    elbow_meas_tau,
    shoulder_tau_controller,
    elbow_tau_controller,
    shoulder_filtered_vel=None,
    elbow_filtered_vel=None,
    shoulder_des_time=None,
    shoulder_des_pos=None,
    shoulder_des_vel=None,
    shoulder_des_tau=None,
    elbow_des_time=None,
    elbow_des_pos=None,
    elbow_des_vel=None,
    elbow_des_tau=None,
    shoulder_fric_tau=None,
    elbow_fric_tau=None,
    error=None,
    show=True,
):
    # position plot of elbow
    print("plotting started")
    print("plotting elbow position")
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], elbow_meas_pos[:index], label="measured position")
    if not (elbow_des_pos is None):
        plt.plot(elbow_des_time, elbow_des_pos, label="desired position")
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Elbow Position (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, "elbow_swingup_pos.pdf"))

    # velocity plot of elbow
    print("plotting elbow velocity")
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], elbow_meas_vel[:index], label="measured velocity")
    if not (elbow_filtered_vel is None):
        plt.plot(
            meas_time[:index], elbow_filtered_vel[:index], label="filtered velocity"
        )
    if not (elbow_des_vel is None):
        plt.plot(elbow_des_time, elbow_des_vel, label="desired velocity")
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad)")
    plt.title("Elbow Velocity (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, "elbow_swingup_vel.pdf"))

    # torque plot for elbow
    print("plotting elbow torque")
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], elbow_meas_tau[:index], label="total commanded torque")
    plt.plot(meas_time[:index], elbow_tau_controller[:index], label="controller torque")
    if not (elbow_des_tau is None):
        plt.plot(elbow_des_time, elbow_des_tau, label="desired torque")
    if not (elbow_fric_tau is None):
        plt.plot(
            meas_time[:index], elbow_fric_tau[:index], label="friction comp. torque"
        )
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Elbow Torque (Nm) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, "elbow_swingup_torque.pdf"))

    # position plot of shoulder
    print("plotting shoulder position")
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], shoulder_meas_pos[:index], label="measured position")
    if not (shoulder_des_pos is None):
        plt.plot(
            shoulder_des_time,
            shoulder_des_pos,
            label="desired position",
        )
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Shoulder Position (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, "shoulder_swingup_pos.pdf"))

    # velocity plot of shoulder
    print("plotting shoulder velocity")
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], shoulder_meas_vel[:index], label="measured velocity")
    if not (shoulder_filtered_vel is None):
        plt.plot(
            meas_time[:index], shoulder_filtered_vel[:index], label="filtered velocity"
        )
    if not (shoulder_des_vel is None):
        plt.plot(
            shoulder_des_time,
            shoulder_des_vel,
            label="desired velocity",
        )
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad)")
    plt.title("Shoulder Velocity (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, "shoulder_swingup_vel.pdf"))

    # torque plot for shoulder
    print("plotting shoulder torque")
    fig, ax = plt.subplots()
    plt.plot(
        meas_time[:index], shoulder_meas_tau[:index], label="total commanded torque"
    )
    plt.plot(
        meas_time[:index], shoulder_tau_controller[:index], label="controller torque"
    )
    if not (shoulder_des_tau is None):
        plt.plot(shoulder_des_time, shoulder_des_tau, label="desired torque")
    if not (shoulder_fric_tau is None):
        plt.plot(
            meas_time[:index], shoulder_fric_tau[:index], label="friction comp. torque"
        )
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Shoulder Torque (Nm) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, "shoulder_swingup_torque.pdf"))

    if show:
        plt.show()
    plt.close()


# def plot_figure_single(save_dir,
#                        date,
#                        index,
#                        meas_time,
#                        shoulder_meas_pos,
#                        shoulder_meas_vel,
#                        shoulder_meas_tau,
#                        elbow_meas_pos,
#                        elbow_meas_vel,
#                        elbow_meas_tau,
#                        shoulder_tau_controller,
#                        elbow_tau_controller,
#                        shoulder_filtered_vel=None,
#                        elbow_filtered_vel=None,
#                        shoulder_des_time=None,
#                        shoulder_des_pos=None,
#                        shoulder_des_vel=None,
#                        shoulder_des_tau=None,
#                        elbow_des_time=None,
#                        elbow_des_pos=None,
#                        elbow_des_vel=None,
#                        elbow_des_tau=None,
#                        shoulder_fric_tau=None,
#                        elbow_fric_tau=None,
#                        error=None):
#
#     fig, ax = plt.subplots(3,
#                            1,
#                            figsize=(18, 9),
#                            sharex="all")
#
#     # position plot
#     ax[0].plot(meas_time[:index], shoulder_meas_pos[:index], label="shoulder measured position", color="blue")
#     ax[0].plot(meas_time[:index], elbow_meas_pos[:index], label='elbow measured position', color="green")
#     if not (shoulder_des_pos is None):
#         ax[0].plot(shoulder_des_time, shoulder_des_pos, label="shoulder desired position", color="lightblue")
#     if not (elbow_des_pos is None):
#         ax[0].plot(elbow_des_time, elbow_des_pos, label="elbow desired position", color="lightgreen")
#     ax[0].set_ylabel("Position (rad)")
#     ax[0].legend()
#
#     # velocity plot
#     ax[1].plot(meas_time[:index], shoulder_meas_vel[:index], label="shoulder measured velocity", color="blue")
#     if not (shoulder_filtered_vel is None):
#         ax[1].plot(meas_time[:index], shoulder_filtered_vel[:index], label="shoulder filtered velocity", color="darkviolet")
#     ax[1].plot(meas_time[:index], elbow_meas_vel[:index], label="elbow measured velocity", color="green")
#     if not (elbow_filtered_vel is None):
#         ax[1].plot(meas_time[:index], elbow_filtered_vel[:index], label="elbow filtered velocity", color="orange")
#     if not (shoulder_des_vel is None):
#         ax[1].plot(shoulder_des_time, shoulder_des_vel, label="shoulder desired velocity", color="lightblue")
#     if not (elbow_des_vel is None):
#         ax[1].plot(elbow_des_time, elbow_des_vel, label="elbow desired velocity", color="lightgreen")
#     ax[1].set_ylabel("Velocity (rad)")
#     ax[1].legend()
#
#     # torque plot for elbow
#     print('plotting elbow torque')
#     ax[2].plot(meas_time[:index], shoulder_meas_tau[:index], label="shoulder total commanded torque", color="blue")
#     ax[2].plot(meas_time[:index], shoulder_tau_controller[:index], label="shoulder controller torque", color="aqua")
#     ax[2].plot(meas_time[:index], elbow_meas_tau[:index], label="elbow total commanded torque", color="green")
#     ax[2].plot(meas_time[:index], elbow_tau_controller[:index], label="elbow controller torque", color="lime")
#     if not (shoulder_des_tau is None):
#         ax[2].plot(shoulder_des_time, shoulder_des_tau, label="shoulder desired torque", color="lightblue")
#     if not (shoulder_fric_tau is None):
#         ax[2].plot(meas_time[:index], shoulder_fric_tau[:index], label="shoulder friction comp. torque", color="darkviolet")
#     if not (elbow_des_tau is None):
#         ax[2].plot(elbow_des_time, elbow_des_tau, label="elbow desired torque", color="lightgreen")
#     if not (elbow_fric_tau is None):
#         ax[2].plot(meas_time[:index], elbow_fric_tau[:index], label="elbow friction comp. torque", color="orange")
#     ax[2].set_xlabel("Time (s)")
#     ax[2].set_ylabel("Torque (Nm)")
#     ax[2].legend()
#
#     plt.savefig(os.path.join(save_dir, f'{date}_combiplot.pdf'))
#     plt.show()
#
