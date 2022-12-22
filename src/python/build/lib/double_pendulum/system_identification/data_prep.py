import numpy as np
from scipy import signal

from double_pendulum.utils.filters.low_pass import lowpass_filter
from double_pendulum.utils.filters.butterworth import butterworth_filter


def smooth_data(t,
                shoulder_pos,
                shoulder_vel,
                shoulder_trq,
                elbow_pos,
                elbow_vel,
                elbow_trq,
                filt="butterworth"):

    if filt == "butterworth":
        Wn = 0.02
        filtered_shoulder_vel = butterworth_filter(shoulder_vel, 3, Wn)
        filtered_elbow_vel = butterworth_filter(elbow_vel, 3, Wn)
        filtered_shoulder_trq = butterworth_filter(shoulder_trq, 3, Wn)
        filtered_elbow_trq = butterworth_filter(elbow_trq, 3, Wn)

        # compute acceleration from positions and filter 2x
        vel1 = np.gradient(shoulder_pos, t)
        vel2 = np.gradient(elbow_pos, t)

        vel1 = butterworth_filter(vel1, 3, Wn)
        vel2 = butterworth_filter(vel2, 3, Wn)

        filtered_shoulder_acc = np.gradient(vel1, t)
        filtered_elbow_acc = np.gradient(vel2, t)

        filtered_shoulder_acc = butterworth_filter(filtered_shoulder_acc, 3, Wn)
        filtered_elbow_acc = butterworth_filter(filtered_elbow_acc, 3, Wn)

    elif filt == "lowpass":
        filtered_shoulder_vel = lowpass_filter(shoulder_vel, 0.3)
        filtered_elbow_vel = lowpass_filter(elbow_vel, 0.3)
        filtered_shoulder_trq = lowpass_filter(shoulder_trq, 0.3)
        filtered_elbow_trq = lowpass_filter(elbow_trq, 0.3)

        # compute acceleration from positions and filter 2x
        vel1 = np.gradient(shoulder_pos, t)
        vel2 = np.gradient(elbow_pos, t)

        vel1 = lowpass_filter(vel1, 0.3)
        vel2 = lowpass_filter(vel2, 0.3)

        filtered_shoulder_acc = np.gradient(vel1, t)
        filtered_elbow_acc = np.gradient(vel2, t)

        filtered_shoulder_acc = lowpass_filter(filtered_shoulder_acc, 0.5)
        filtered_elbow_acc = lowpass_filter(filtered_elbow_acc, 0.5)
    else:
        filtered_shoulder_vel = shoulder_vel
        filtered_elbow_vel = elbow_vel
        filtered_shoulder_trq = shoulder_trq
        filtered_elbow_trq = elbow_trq
        
        vel1 = np.gradient(shoulder_pos, t)
        vel2 = np.gradient(elbow_pos, t)

        filtered_shoulder_acc = np.gradient(vel1, t)
        filtered_elbow_acc = np.gradient(vel2, t)

    return (t,
            shoulder_pos,
            elbow_pos,
            filtered_shoulder_vel,
            filtered_elbow_vel,
            filtered_shoulder_acc,
            filtered_elbow_acc,
            filtered_shoulder_trq,
            filtered_elbow_trq)
