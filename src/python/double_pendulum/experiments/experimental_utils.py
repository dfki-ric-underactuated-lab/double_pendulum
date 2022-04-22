import os
import math
import numpy as np
# from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def yb_friction_matrix(dq_vec):

    y_11 = math.atan(100*dq_vec[0])
    y_12 = dq_vec[0]

    y_23 = math.atan(100*dq_vec[1])
    y_24 = dq_vec[1]

    yb_fric = np.array([[y_11, y_12, 0, 0],
                        [0, 0, y_23, y_24]])
    return yb_fric


# def read_data():
#     MAIN_DIR = os.path.realpath(os.curdir)
#     print("Working DIR:", MAIN_DIR)
#     INPUT_FILENAME = "pendubot_swingup_1000Hz.csv"
#     INPUT_PATH = "/"
#     data = pd.read_csv(MAIN_DIR + INPUT_PATH + INPUT_FILENAME)
#     n = len(data)
#     return MAIN_DIR, INPUT_PATH, INPUT_FILENAME, data, n


def wrap_angle_pi2pi(angle):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


# def prepare_data(data, n):
#     # Prepare data for creating TVLQR
#     shoulder_pos_traj = data["shoulder_pos"]
#     shoulder_vel_traj = data["shoulder_vel"]
#     shoulder_tau_traj = data["shoulder_torque"]
#     elbow_pos_traj = data["elbow_pos"]
#     elbow_vel_traj = data["elbow_vel"]
#     elbow_tau_traj = data["elbow_torque"]
#     time_traj = data["time"]
#     # Creating empty arrays for sensor data measurement
#     shoulder_meas_pos = np.zeros(n)
#     shoulder_meas_vel = np.zeros(n)
#     shoulder_meas_tau = np.zeros(n)
#     elbow_meas_pos = np.zeros(n)
#     elbow_meas_vel = np.zeros(n)
#     elbow_meas_tau = np.zeros(n)
#     abs_error_state = np.zeros(n)
#     shoulder_on = np.zeros(n)
#     meas_time = np.zeros(n)
#     # transmission of the motor
#     gear_ratio = 1
#     rad2outputrev = gear_ratio / (2 * np.pi)
#     # torque in Nm on the motor side before the gear transmission
#     shoulder_tau_in = shoulder_tau_traj
#     elbow_tau_in = elbow_tau_traj
#     # time difference between two consecutive data points
#     dt = time_traj[2] - time_traj[1]
# 
#     return (
#         shoulder_pos_traj, shoulder_vel_traj, shoulder_tau_traj,
#         elbow_pos_traj, elbow_vel_traj, elbow_tau_traj,
#         time_traj,
#         shoulder_meas_pos, shoulder_meas_vel, shoulder_meas_tau,
#         elbow_meas_pos, elbow_meas_vel, elbow_meas_tau,
#         meas_time,
#         gear_ratio, rad2outputrev, shoulder_tau_in, elbow_tau_in,
#         dt, abs_error_state, shoulder_on)
# 
# 
# def prepare_empty_data(n):
#     # Creating empty arrays for sensor data measurement
#     shoulder_meas_pos = np.zeros(n)
#     shoulder_meas_vel = np.zeros(n)
#     shoulder_meas_tau = np.zeros(n)
#     shoulder_on = np.zeros(n)
#     elbow_meas_pos = np.zeros(n)
#     elbow_meas_vel = np.zeros(n)
#     elbow_meas_tau = np.zeros(n)
#     meas_time = np.zeros(n)
#     # transmission of the motor
#     gear_ratio = 1
#     rad2outputrev = gear_ratio / (2 * np.pi)
#     # torque in Nm on the motor side before the gear transmission
# 
#     return (shoulder_meas_pos,
#             shoulder_meas_vel,
#             shoulder_meas_tau,
#             elbow_meas_pos,
#             elbow_meas_vel,
#             elbow_meas_tau,
#             meas_time,
#             gear_ratio,
#             rad2outputrev,
#             shoulder_on)

def plot_figure(save_dir,
                date,
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
                shoulder_filtered_vel,
                elbow_filtered_vel,
                shoulder_des_pos=None,
                shoulder_des_vel=None,
                shoulder_des_tau=None,
                elbow_des_pos=None,
                elbow_des_vel=None,
                elbow_des_tau=None,
                shoulder_fric_tau=None,
                elbow_fric_tau=None,
                error=None):

    # position plot of elbow
    print('plotting started')
    print('plotting elbow position')
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], elbow_meas_pos[:index], label='measured position')
    if not (elbow_des_pos is None):
        plt.plot(meas_time[:index], elbow_des_pos[:index], label="desired position")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
    #                 facecolor='green', alpha=0.5, transform=trans)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
    #                 facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Elbow Position (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_elbow_swingup_pos.pdf'))

    # velocity plot of elbow
    print('plotting elbow velocity')
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], elbow_meas_vel[:index], label="measured velocity")
    plt.plot(meas_time[:index], elbow_filtered_vel[:index], label="filtered velocity")
    if not (elbow_des_vel is None):
        plt.plot(meas_time[:index], elbow_des_vel[:index], label="desired velocity")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
    #                 facecolor='green', alpha=0.5, transform=trans)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
    #                 facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad)")
    plt.title("Elbow Velocity (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_elbow_swingup_vel.pdf'))

    # torque plot for elbow
    print('plotting elbow torque')
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], elbow_meas_tau[:index], label="total commanded torque")
    plt.plot(meas_time[:index], elbow_tau_controller[:index], label="controller torque")
    if not (elbow_des_tau is None):
        plt.plot(meas_time[:index], elbow_des_tau[:index], label="desired torque")
    if not (elbow_fric_tau is None):
        plt.plot(meas_time[:index], elbow_fric_tau[:index], label="friction comp. torque")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
    #                 facecolor='green', alpha=0.5, transform=trans)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
    #                 facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Elbow Torque (Nm) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_elbow_swingup_torque.pdf'))

    # position plot of shoulder
    print('plotting shoulder position')
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], shoulder_meas_pos[:index], label="measured position")
    if not (shoulder_des_pos is None):
        plt.plot(meas_time[:index], shoulder_des_pos[:index], label="desired position")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
    #                 facecolor='green', alpha=0.5, transform=trans)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
    #                 facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Shoulder Position (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_shoulder_swingup_pos.pdf'))

    # velocity plot of shoulder
    print('plotting shoulder velocity')
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], shoulder_meas_vel[:index], label="measured velocity")
    plt.plot(meas_time[:index], shoulder_filtered_vel[:index], label="filtered velocity")
    if not (shoulder_des_vel is None):
        plt.plot(meas_time[:index], shoulder_des_vel[:index], label="desired velocity")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
    #                 facecolor='green', alpha=0.5, transform=trans)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
    #                 facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad)")
    plt.title("Shoulder Velocity (rad) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_shoulder_swingup_vel.pdf'))

    # torque plot for shoulder
    print('plotting shoulder torque')
    fig, ax = plt.subplots()
    plt.plot(meas_time[:index], shoulder_meas_tau[:index], label="total commanded torque")
    plt.plot(meas_time[:index], shoulder_tau_controller[:index], label="controller torque")
    if not (shoulder_des_tau is None):
        plt.plot(meas_time[:index], shoulder_des_tau[:index], label="desired torque")
    if not (shoulder_fric_tau is None):
        plt.plot(meas_time[:index], shoulder_fric_tau[:index], label="friction comp. torque")
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
    #                 facecolor='green', alpha=0.5, transform=trans)
    # ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
    #                 facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Shoulder Torque (Nm) vs Time (s)")
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_shoulder_swingup_torque.pdf'))

    plt.show()

def plot_figure_single(save_dir,
                       date,
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
                       shoulder_filtered_vel,
                       elbow_filtered_vel,
                       shoulder_des_pos=None,
                       shoulder_des_vel=None,
                       shoulder_des_tau=None,
                       elbow_des_pos=None,
                       elbow_des_vel=None,
                       elbow_des_tau=None,
                       shoulder_fric_tau=None,
                       elbow_fric_tau=None,
                       error=None):

    fig, ax = plt.subplots(3,
                           1,
                           figsize=(18, 9),
                           sharex="all")

    # position plot
    ax[0].plot(meas_time[:index], shoulder_meas_pos[:index], label="shoulder measured position", color="blue")
    ax[0].plot(meas_time[:index], elbow_meas_pos[:index], label='elbow measured position', color="green")
    if not (shoulder_des_pos is None):
        ax[0].plot(meas_time[:index], shoulder_des_pos[:index], label="shoulder desired position", color="lightblue")
    if not (elbow_des_pos is None):
        ax[0].plot(meas_time[:index], elbow_des_pos[:index], label="elbow desired position", color="lightgreen")
    ax[0].set_ylabel("Position (rad)")
    ax[0].legend()

    # velocity plot
    ax[1].plot(meas_time[:index], shoulder_meas_vel[:index], label="shoulder measured velocity", color="blue")
    ax[1].plot(meas_time[:index], shoulder_filtered_vel[:index], label="shoulder filtered velocity", color="darkviolet")
    ax[1].plot(meas_time[:index], elbow_meas_vel[:index], label="elbow measured velocity", color="green")
    ax[1].plot(meas_time[:index], elbow_filtered_vel[:index], label="elbow filtered velocity", color="orange")
    if not (shoulder_des_vel is None):
        ax[1].plot(meas_time[:index], shoulder_des_vel[:index], label="shoulder desired velocity", color="lightblue")
    if not (elbow_des_vel is None):
        ax[1].plot(meas_time[:index], elbow_des_vel[:index], label="elbow desired velocity", color="lightgreen")
    ax[1].set_ylabel("Velocity (rad)")
    ax[1].legend()

    # torque plot for elbow
    print('plotting elbow torque')
    ax[2].plot(meas_time[:index], shoulder_meas_tau[:index], label="shoulder total commanded torque", color="blue")
    ax[2].plot(meas_time[:index], shoulder_tau_controller[:index], label="shoulder controller torque", color="aqua")
    ax[2].plot(meas_time[:index], elbow_meas_tau[:index], label="elbow total commanded torque", color="green")
    ax[2].plot(meas_time[:index], elbow_tau_controller[:index], label="elbow controller torque", color="lime")
    if not (shoulder_des_tau is None):
        ax[2].plot(meas_time[:index], shoulder_des_tau[:index], label="shoulder desired torque", color="lightblue")
    if not (shoulder_fric_tau is None):
        ax[2].plot(meas_time[:index], shoulder_fric_tau[:index], label="shoulder friction comp. torque", color="darkviolet")
    if not (elbow_des_tau is None):
        ax[2].plot(meas_time[:index], elbow_des_tau[:index], label="elbow desired torque", color="lightgreen")
    if not (elbow_fric_tau is None):
        ax[2].plot(meas_time[:index], elbow_fric_tau[:index], label="elbow friction comp. torque", color="orange")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Torque (Nm)")
    ax[2].legend()

    plt.savefig(os.path.join(save_dir, f'{date}_combiplot.pdf'))
    plt.show()

def save_data(save_dir,
              date,
              shoulder_meas_pos,
              shoulder_meas_vel,
              shoulder_meas_tau,
              elbow_meas_pos,
              elbow_meas_vel,
              elbow_meas_tau,
              meas_time):
    print("Saving data to .csv files.")
    measured_csv_data = np.array([np.array(meas_time),
                                  np.array(shoulder_meas_pos),
                                  np.array(shoulder_meas_vel),
                                  np.array(shoulder_meas_tau),
                                  np.array(elbow_meas_pos),
                                  np.array(elbow_meas_vel),
                                  np.array(elbow_meas_tau)]).T
    np.savetxt(os.path.join(save_dir, f'{date}_measured.csv'),
               measured_csv_data,
               delimiter=',',
               # header="meas_time,shoulder_meas_pos,shoulder_meas_vel,shoulder_meas_tau,elbow_meas_pos,elbow_meas_vel,elbow_meas_tau",
               header="time,shoulder_pos,shoulder_vel,shoulder_torque,elbow_pos,elbow_vel,elbow_tau",
               comments="")
    print("CSV file saved\n")


# def setZeroPosition(motor, initPos):
#     pos = initPos
#     while abs(np.rad2deg(pos)) > 0.5:
#         pos, vel, curr = motor.set_zero_position()
#         print("Position: {}, Velocity: {}, Torque: {}".format(
#             np.rad2deg(pos), np.rad2deg(vel), curr))

def setZeroPosition(motor, initPos, initVel, initTau):
    pos = np.copy(initPos)
    vel = np.copy(initVel)
    tau = np.copy(initTau)
    while (abs(np.rad2deg(pos)) > 0.5 or abs(np.rad2deg(vel)) > 0.5 or abs(tau) > 0.1):
        motor.set_zero_position()
        pos, vel, tau = motor.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                tau))

def rad2rev(angle_in_radians):
    return angle_in_radians * (1 / (2 * np.pi))


def rev2rad(angle_in_revolution):
    return angle_in_revolution * (2 * np.pi)

