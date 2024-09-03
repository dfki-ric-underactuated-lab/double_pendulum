import os
import time
import numpy as np

# from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def save_data(
    save_dir,
    date,
    shoulder_meas_pos,
    shoulder_meas_vel,
    shoulder_meas_tau,
    elbow_meas_pos,
    elbow_meas_vel,
    elbow_meas_tau,
    meas_time,
):
    """
    save data to csv file.
    Deprecated. One should use double_pendulum.utils.save_trajectory instead.
    """
    print("Saving data to .csv files.")
    measured_csv_data = np.array(
        [
            np.array(meas_time),
            np.array(shoulder_meas_pos),
            np.array(elbow_meas_pos),
            np.array(shoulder_meas_vel),
            np.array(elbow_meas_vel),
            np.array(shoulder_meas_tau),
            np.array(elbow_meas_tau),
        ]
    ).T
    np.savetxt(
        os.path.join(save_dir, f"{date}_measured.csv"),
        measured_csv_data,
        delimiter=",",
        # header="meas_time,shoulder_meas_pos,shoulder_meas_vel,shoulder_meas_tau,elbow_meas_pos,elbow_meas_vel,elbow_meas_tau",
        header="time,pos1,pos2,vel1,vel2,tau1,tau2",
        comments="",
    )
    print("CSV file saved\n")


def setZeroPosition(motor, motor_direction=1.0):
    """
    Set the zero position for a tmotor.

    Parameters
    ----------
    motor : motor_driver.canmotorlib.CanMotorController
        motor whose position will be initialized
    """

    print("Setting zero position of motor...")

    pos, vel, tau = motor.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
    pos *= motor_direction
    vel *= motor_direction
    tau *= motor_direction
    while abs(np.rad2deg(pos)) > 0.5 or abs(np.rad2deg(vel)) > 0.5 or abs(tau) > 0.1:
        motor.set_zero_position()
        pos, vel, tau = motor.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
        pos *= motor_direction
        vel *= motor_direction
        tau *= motor_direction
        print(
            "Position: {}, Velocity: {}, Torque: {}".format(
                np.rad2deg(pos), np.rad2deg(vel), tau
            )
        )


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


def go_to_zero(motor1, motor2, motor_directions=[1.0, 1.0]):
    print("Moving back to initial configuration...")
    try:
        counter = 0
        pos_epsilon = 0.05
        vel_epsilon = 0.1
        pos1, vel1, tau1 = motor1.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
        pos2, vel2, tau2 = motor2.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)

        # correction for motor axis directions
        pos1 *= motor_directions[0]
        vel1 *= motor_directions[0]
        tau1 *= motor_directions[0]
        pos2 *= motor_directions[1]
        vel2 *= motor_directions[1]
        tau2 *= motor_directions[1]

        # multi rotation check
        goal1 = np.round(pos1 / (2.0 * np.pi)) * 2.0 * np.pi
        goal2 = np.round(pos2 / (2.0 * np.pi)) * 2.0 * np.pi

        while (
            np.abs(pos1 - goal1) > pos_epsilon
            or np.abs(pos2 - goal2) > pos_epsilon
            or np.abs(vel1) > vel_epsilon
            or np.abs(vel2) > vel_epsilon
        ):
            start_loop = time.time()

            if np.abs(pos1) > 0.1:
                next_pos1 = pos1 - np.sign(pos1 - goal1) * 0.1
            else:
                next_pos1 = goal1
            if np.abs(pos2) > 0.1:
                next_pos2 = pos2 - np.sign(pos2 - goal2) * 0.1
            else:
                next_pos2 = goal2

            next_pos1 *= motor_directions[0]
            next_pos2 *= motor_directions[1]

            (
                pos1,
                vel1,
                shoulder_tau,
            ) = motor1.send_rad_command(next_pos1, 0.0, 5.0, 1.0, 0.0)
            (
                pos2,
                vel2,
                elbow_tau,
            ) = motor2.send_rad_command(next_pos2, 0.0, 5.0, 1.0, 0.0)

            # correction for motor axis directions
            pos1 *= motor_directions[0]
            vel1 *= motor_directions[0]
            tau1 *= motor_directions[0]
            pos2 *= motor_directions[1]
            vel2 *= motor_directions[1]
            tau2 *= motor_directions[1]

            while time.time() - start_loop < 0.01:
                pass
            counter += 1
            if counter > 1000:
                break
            print(pos1, pos2, vel1, vel2, end="\r")
    except TypeError:
        pass
    print("\nDone")


def slow_down_motors(motr1, motor2):
    pass


def enable_motor(motor):
    try:
        print("Enabling motor...")
        _ = motor.enable_motor()
        print("Motor enabled")

    except TypeError:
        pass


def disable_motor(motor):
    try:
        print("Disabling motor...")
        _ = motor.disable_motor()
        print("Motor disabled")

    except TypeError:
        pass


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
