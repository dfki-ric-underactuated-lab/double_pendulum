import numpy as np
from scipy import signal


def smooth_data_butter(t,
                       shoulder_pos,
                       shoulder_vel,
                       shoulder_trq,
                       elbow_pos,
                       elbow_vel,
                       elbow_trq):
    """
    velocity data filter
    """
    b_shoulder_vel, a_shoulder_vel = signal.butter(3, 0.2)
    filtered_shoulder_vel = signal.filtfilt(b_shoulder_vel,
                                            a_shoulder_vel,
                                            shoulder_vel)
    b_elbow_vel, a_elbow_vel = signal.butter(3, 0.2)
    filtered_elbow_vel = signal.filtfilt(b_elbow_vel, a_elbow_vel, elbow_vel)

    """
    torque data filter
    """
    b_shoulder_trq, a_shoulder_trq = signal.butter(3, 0.1)
    filtered_shoulder_trq = signal.filtfilt(b_shoulder_trq, a_shoulder_trq, shoulder_trq)
    b_elbow_trq, a_elbow_trq = signal.butter(3, 0.1)
    filtered_elbow_trq = signal.filtfilt(b_elbow_trq, a_elbow_trq, elbow_trq)

    """
    acceleration data filter
    """
    shoulder_vel3 = np.gradient(shoulder_pos, t)
    b_shoulder_vel, a_shoulder_vel = signal.butter(3, 0.5)
    shoulder_vel3 = signal.filtfilt(b_shoulder_vel, a_shoulder_vel,
                                    shoulder_vel3)
    shoulder_acc_1 = np.gradient(shoulder_vel3, t)
    b_shoulder_acc, a_shoulder_acc = signal.butter(3, 0.1)
    filtered_shoulder_acc = signal.filtfilt(b_shoulder_acc,
                                            a_shoulder_acc,
                                            shoulder_acc_1)

    elbow_vel3 = np.gradient(elbow_pos, t)
    b_elbow_vel, a_elbow_vel = signal.butter(3, 0.5)
    elbow_vel3 = signal.filtfilt(b_elbow_vel, a_elbow_vel, elbow_vel3)
    elbow_acc_1 = np.gradient(elbow_vel3, t)
    b_elbow_acc, a_elbow_acc = signal.butter(3, 0.1)
    filtered_elbow_acc = signal.filtfilt(b_elbow_acc, a_elbow_acc, elbow_acc_1)

    return (t,
            shoulder_pos,
            elbow_pos,
            filtered_shoulder_vel,
            filtered_elbow_vel,
            filtered_shoulder_acc,
            filtered_elbow_acc,
            filtered_shoulder_trq,
            filtered_elbow_trq)
