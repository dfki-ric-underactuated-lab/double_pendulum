import os
from datetime import datetime
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

    yb_fric = np.array([[y_11, y_12,0,0],
                        [0, 0, y_23, y_24]])
    return yb_fric

def read_data():
    MAIN_DIR = os.path.realpath(os.curdir)
    print("Working DIR:", MAIN_DIR)
    INPUT_FILENAME = "pendubot_swingup_1000Hz.csv"
    INPUT_PATH = "/"
    data = pd.read_csv(MAIN_DIR + INPUT_PATH + INPUT_FILENAME)
    n = len(data)
    return MAIN_DIR, INPUT_PATH, INPUT_FILENAME, data, n


def wrap_angle_pi2pi(angle):
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def prepare_data(data, n):
    # Prepare data for creating TVLQR
    shoulder_pos_traj = data["shoulder_pos"]
    shoulder_vel_traj = data["shoulder_vel"]
    shoulder_tau_traj = data["shoulder_torque"]
    elbow_pos_traj = data["elbow_pos"]
    elbow_vel_traj = data["elbow_vel"]
    elbow_tau_traj = data["elbow_torque"]
    time_traj = data["time"]
    # Creating empty arrays for sensor data measurement
    shoulder_meas_pos = np.zeros(n)
    shoulder_meas_vel = np.zeros(n)
    shoulder_meas_tau = np.zeros(n)
    elbow_meas_pos = np.zeros(n)
    elbow_meas_vel = np.zeros(n)
    elbow_meas_tau = np.zeros(n)
    abs_error_state = np.zeros(n)
    shoulder_on = np.zeros(n)
    meas_time = np.zeros(n)
    # transmission of the motor
    gear_ratio = 1
    rad2outputrev = gear_ratio / (2 * np.pi)
    # torque in Nm on the motor side before the gear transmission
    shoulder_tau_in = shoulder_tau_traj
    elbow_tau_in = elbow_tau_traj
    # time difference between two consecutive data points
    dt = time_traj[2] - time_traj[1]

    return (
        shoulder_pos_traj, shoulder_vel_traj, shoulder_tau_traj, elbow_pos_traj, elbow_vel_traj, elbow_tau_traj,
        time_traj,
        shoulder_meas_pos, shoulder_meas_vel, shoulder_meas_tau, elbow_meas_pos, elbow_meas_vel, elbow_meas_tau,
        meas_time,
        gear_ratio, rad2outputrev, shoulder_tau_in, elbow_tau_in, dt, abs_error_state,
        shoulder_on)  # , shoulder_pos_out, shoulder_vel_out, elbow_pos_out, elbow_vel_out

def prepare_empty_data(n):
    # Creating empty arrays for sensor data measurement
    shoulder_meas_pos = np.zeros(n)
    shoulder_meas_vel = np.zeros(n)
    shoulder_meas_tau = np.zeros(n)
    shoulder_on = np.zeros(n)
    elbow_meas_pos = np.zeros(n)
    elbow_meas_vel = np.zeros(n)
    elbow_meas_tau = np.zeros(n)
    meas_time = np.zeros(n)
    # transmission of the motor
    gear_ratio = 1
    rad2outputrev = gear_ratio / (2 * np.pi)
    # torque in Nm on the motor side before the gear transmission

    return (
        shoulder_meas_pos,
        shoulder_meas_vel,
        shoulder_meas_tau,
        elbow_meas_pos,
        elbow_meas_vel,
        elbow_meas_tau,
        meas_time,
        gear_ratio,
        rad2outputrev,
        shoulder_on)

def plot_figure(save_dir,
        date,
        shoulder_meas_pos,
        shoulder_meas_vel,
        shoulder_meas_tau,
        elbow_meas_pos,
        elbow_meas_vel,
        elbow_meas_tau,
        meas_time,
        shoulder_on):

    # position plot of elbow
    print('plotting started')
    print('plotting elbow position')
    fig, ax = plt.subplots()
    plt.plot(meas_time, elbow_meas_pos)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
                    facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
                    facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Elbow Position (rad) vs Time (s)")
    plt.legend(['position_measured'])
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_elbow_swingup_pos.pdf'))
    # plt.show()

    print('plotting elbow velocity')
    # velocity plot of elbow
    fig, ax = plt.subplots()
    plt.plot(meas_time, elbow_meas_vel)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
                    facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
                    facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad)")
    plt.title("Elbow Velocity (rad) vs Time (s)")
    plt.legend(['velocity_measured'])
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_elbow_swingup_vel.pdf'))
    # plt.show()

    print('plotting elbow torque')
    # torque plot for elbow
    fig, ax = plt.subplots()
    plt.plot(meas_time, elbow_meas_tau)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
                    facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
                    facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Elbow Torque (Nm) vs Time (s)")
    plt.legend(['Measured Torque'])
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_elbow_swingup_torque.pdf'))
    # plt.show()

    # position plot of shoulder
    print('plotting shoulder position')
    fig, ax = plt.subplots()
    plt.plot(meas_time, shoulder_meas_pos)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
                    facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
                    facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.title("Shoulder Position (rad) vs Time (s)")
    plt.legend(['position_measured'])
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_shoulder_swingup_pos.pdf'))
    # plt.show()

    # velocity plot of shoulder
    print('plotting shoulder velocity')
    fig, ax = plt.subplots()
    plt.plot(meas_time, shoulder_meas_vel)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
                    facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
                    facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad)")
    plt.title("Shoulder Velocity (rad) vs Time (s)")
    plt.legend(['velocity_measured'])
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_shoulder_swingup_vel.pdf'))
    # plt.show()
    # torque plot for shoulder
    print('plotting shoulder torque')
    fig, ax = plt.subplots()
    plt.plot(meas_time, shoulder_meas_tau)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 1,
                    facecolor='green', alpha=0.5, transform=trans)
    ax.fill_between(meas_time, 0, 1, where=shoulder_on == 0,
                    facecolor='white', alpha=1, transform=trans)
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Shoulder Torque (Nm) vs Time (s)")
    plt.legend(['Measured Torque'])
    plt.draw()
    plt.savefig(os.path.join(save_dir, f'{date}_shoulder_swingup_torque.pdf'))
    plt.show()
    return date


def save_data(save_dir,
              date,
              shoulder_meas_pos,
              shoulder_meas_vel,
              shoulder_meas_tau,
              elbow_meas_pos,
              elbow_meas_vel,
              elbow_meas_tau,
              meas_time,
              shoulder_on):
    print("Saving data to .csv files.")
    measured_csv_data = np.array([np.array(meas_time),
                                  np.array(shoulder_meas_pos),
                                  np.array(shoulder_meas_vel),
                                  np.array(shoulder_meas_tau),
                                  np.array(elbow_meas_pos),
                                  np.array(elbow_meas_vel),
                                  np.array(elbow_meas_tau),
                                  np.array(shoulder_on)]).T
    np.savetxt(os.path.join(save_dir, f'{date}_measured.csv'),
               measured_csv_data,
               delimiter=',',
               header="meas_time,shoulder_meas_pos,shoulder_meas_vel,shoulder_meas_tau,elbow_meas_pos,elbow_meas_vel,elbow_meas_tau,shoulder_on",
               comments="")
    print("CSV file saved\n")


def setZeroPosition(motor, initPos):
    pos = initPos
    while abs(np.rad2deg(pos)) > 0.5:
        pos, vel, curr = motor.set_zero_position()
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                curr))

