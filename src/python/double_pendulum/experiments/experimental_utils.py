import os
import math
import numpy as np
# from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# def wrap_angle_pi2pi(angle):
#     angle = angle % (2 * np.pi)
#     if angle > np.pi:
#         angle -= 2 * np.pi
#     return angle


def save_data(save_dir,
              date,
              shoulder_meas_pos,
              shoulder_meas_vel,
              shoulder_meas_tau,
              elbow_meas_pos,
              elbow_meas_vel,
              elbow_meas_tau,
              meas_time):
    """
    save data to csv file.
    Deprecated. One should use double_pendulum.utils.save_trajectory instead.
    """
    print("Saving data to .csv files.")
    measured_csv_data = np.array([np.array(meas_time),
                                  np.array(shoulder_meas_pos),
                                  np.array(elbow_meas_pos),
                                  np.array(shoulder_meas_vel),
                                  np.array(elbow_meas_vel),
                                  np.array(shoulder_meas_tau),
                                  np.array(elbow_meas_tau)]).T
    np.savetxt(os.path.join(save_dir, f'{date}_measured.csv'),
               measured_csv_data,
               delimiter=',',
               #header="meas_time,shoulder_meas_pos,shoulder_meas_vel,shoulder_meas_tau,elbow_meas_pos,elbow_meas_vel,elbow_meas_tau",
               header="time,pos1,pos2,vel1,vel2,tau1,tau2",
               comments="")
    print("CSV file saved\n")

def setZeroPosition(motor, initPos, initVel, initTau):
    """
    Set the zero position for a tmotor.

    Parameters
    ----------
    motor : motor_driver.canmotorlib.CanMotorController
        motor whose position will be initialized
    initPos : float
        initial motor position from sensor readings
    initPos : float
        initial motor velocity from sensor readings
    initPos : float
        initial motor torque from sensor readings
    """
    pos = np.copy(initPos)
    vel = np.copy(initVel)
    tau = np.copy(initTau)
    while (abs(np.rad2deg(pos)) > 0.5 or abs(np.rad2deg(vel)) > 0.5 or abs(tau) > 0.1):
        motor.set_zero_position()
        pos, vel, tau = motor.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
        print("Position: {}, Velocity: {}, Torque: {}".format(np.rad2deg(pos), np.rad2deg(vel),
                                                                tau))

def rad2rev(angle_in_radians):
    """
    Convert radians to revolutions.

    Parameters
    ----------
    angle_in_radians : float
        angle, unit=[rad]

    Returns
    -------
    float
        angle in revolutions
    """
    return angle_in_radians * (1 / (2 * np.pi))


def rev2rad(angle_in_revolution):
    """
    Convert revolutions to radians.

    Parameters
    ----------
    angle_in_revolution : float
        angle, unit=[rev]

    Returns
    -------
    float
        angle in radians
    """
    return angle_in_revolution * (2 * np.pi)

# def setZeroPosition(motor, initPos):
#     pos = initPos
#     while abs(np.rad2deg(pos)) > 0.5:
#         pos, vel, curr = motor.set_zero_position()
#         print("Position: {}, Velocity: {}, Torque: {}".format(
#             np.rad2deg(pos), np.rad2deg(vel), curr))

# def read_data():
#     MAIN_DIR = os.path.realpath(os.curdir)
#     print("Working DIR:", MAIN_DIR)
#     INPUT_FILENAME = "pendubot_swingup_1000Hz.csv"
#     INPUT_PATH = "/"
#     data = pd.read_csv(MAIN_DIR + INPUT_PATH + INPUT_FILENAME)
#     n = len(data)
#     return MAIN_DIR, INPUT_PATH, INPUT_FILENAME, data, n

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

