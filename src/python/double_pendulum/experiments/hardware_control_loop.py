import os
import time
from datetime import datetime
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from double_pendulum.experiments.canmotorlib import CanMotorController
from double_pendulum.experiments.experimental_utils import (yb_friction_matrix,
                                                            prepare_empty_data,
                                                            plot_figure,
                                                            save_data,
                                                            setZeroPosition)

from double_pendulum.experiments.filters.low_pass import online_filter


def run_experiment(controller,
                   dt=0.01,
                   t_final=10,
                   can_port='can0',
                   motor_ids=[8, 9],
                   tau_limit=[4., 4.],
                   friction_compensation=False,
                   friction_terms=[0.093, 0.186, 0.081, 0.0],
                   velocity_filter=None,
                   filter_args={"alpha": 0.3,
                                "kernel_size": 5,
                                "filter_size": 1},
                   save_dir="."):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    n = int(t_final/dt)

    (shoulder_meas_pos,
     shoulder_meas_vel,
     shoulder_meas_tau,
     elbow_meas_pos,
     elbow_meas_vel,
     elbow_meas_tau,
     meas_time,
     gear_ratio,
     rad2outputrev,
     shoulder_on) = prepare_empty_data(n)

    shoulder_filtered_meas_vel = np.copy(shoulder_meas_vel)
    elbow_filtered_meas_vel = np.copy(elbow_meas_vel)

    print("Enabling Motors..")
    # motor_shoulder_id = 0x08  # todo: pass as arguments
    # motor_elbow_id = 0x09     # are these just integers?
    motor_shoulder_id = motor_ids[0]
    motor_elbow_id = motor_ids[1]

    # Create motor controller objects
    motor_shoulder_controller = CanMotorController(can_port, motor_shoulder_id)
    motor_elbow_controller = CanMotorController(can_port, motor_elbow_id)

    (shoulder_pos,
     shoulder_vel,
     shoulder_torque) = motor_shoulder_controller.enable_motor()
    print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
        shoulder_pos, shoulder_vel, shoulder_torque))

    elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()
    print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
        elbow_pos, elbow_vel, elbow_torque))

    print("Setting Shoulder Motor to Zero Position...")
    setZeroPosition(motor_shoulder_controller, shoulder_pos)

    print("Setting Elbow Motor to Zero Position...")
    setZeroPosition(motor_elbow_controller, elbow_pos)

    shoulder_meas_pos[0] = shoulder_pos
    shoulder_meas_vel[0] = shoulder_vel
    shoulder_filtered_meas_vel[0] = shoulder_vel
    elbow_meas_pos[0] = elbow_pos
    elbow_meas_vel[0] = elbow_vel
    elbow_filtered_meas_vel[0] = elbow_vel

    if input('Do you want to proceed for real time execution?(y) ') == 'y':
        print(np.array([shoulder_pos,
                        elbow_pos,
                        shoulder_vel,
                        elbow_vel]))

        # defining running index variables
        index = 0
        t = 0.
        tau_fric = np.zeros(2)

        print("Starting Experiment...")
        # start_time = time.time()
        try:
            while t < t_final:
                start_loop = time.time()

                if velocity_filter == "lowpass":
                    # n_filter = min(index+1, filter_args["filter_size"])

                    # shoulder_filtered_vel = online_filter(
                    #     shoulder_meas_vel[max(index-n_filter, 0):index+1],
                    #     0.3)[-1]
                    # elbow_filtered_vel = online_filter(
                    #     elbow_meas_vel[max(index-n_filter, 0):index+1],
                    #     0.3)[-1]

                    # the lowpass filter needs only the last filtered vel
                    # and the latest vel value
                    sfv = [shoulder_filtered_meas_vel[max(0, index-1)],
                           shoulder_meas_vel[index]]
                    efv = [shoulder_filtered_meas_vel[max(0, index-1)],
                           shoulder_meas_vel[index]]
                    shoulder_filtered_vel = online_filter(
                        sfv, filter_args["alpha"])[-1]
                    elbow_filtered_vel = online_filter(
                        efv, filter_args["alpha"])[-1]
                elif velocity_filter == "medianfilter":
                    n_filter = min(index+1, filter_args["filter_size"])
                    shoulder_filtered_vel = medfilt(
                        shoulder_meas_vel[max(index-n_filter, 0):index+1],
                        kernel_size=filter_args["kernel_size"])[-1]
                    elbow_filtered_vel = medfilt(
                        elbow_meas_vel[max(index-n_filter, 0):index+1],
                        kernel_size=filter_args["kernel_size"])[-1]
                elif velocity_filter == "meanfilter":
                    n_filter = min(index+1, filter_args["filter_size"])
                    shoulder_filtered_vel = np.mean(
                        shoulder_meas_vel[max(index-n_filter, 0):index+1])
                    elbow_filtered_vel = np.mean(
                        elbow_meas_vel[max(index-n_filter, 0):index+1])
                else:
                    shoulder_filtered_vel = shoulder_vel
                    elbow_filtered_vel = elbow_vel

                x = np.array([shoulder_pos,
                              elbow_pos,
                              shoulder_filtered_vel,
                              elbow_filtered_vel])

                # print(x)
                # get control command from controller
                tau_cmd = controller.get_control_output(x)

                # add friction compensation (0 if turned off)
                tau_cmd[0] += tau_fric[0]
                tau_cmd[1] += tau_fric[1]

                # safety command
                tau_cmd[0] = np.clip(tau_cmd[0], -tau_limit[0], tau_limit[0])
                tau_cmd[1] = np.clip(tau_cmd[1], -tau_limit[1], tau_limit[1])

                # Send tau command to motors
                (shoulder_pos,
                 shoulder_vel,
                 shoulder_tau) = motor_shoulder_controller.send_rad_command(
                    0.0, 0.0, 0.0, 0.0, tau_cmd[0])

                (elbow_pos,
                 elbow_vel,
                 elbow_tau) = motor_elbow_controller.send_rad_command(
                    0.0, 0.0, 0.0, 0.0, -tau_cmd[1])  # why the minus sign?

                # friction compensation
                if friction_compensation:
                    # friction_regressor_mat = yb_friction_matrix(
                    #    [shoulder_vel,
                    #     elbow_vel])
                    friction_regressor_mat = yb_friction_matrix(
                        [shoulder_filtered_vel,
                         elbow_filtered_vel])
                    tau_fric = np.dot(friction_regressor_mat,
                                      np.array(friction_terms))

                # store the measured sensor data of
                # position, velocity and torque in each time step
                shoulder_meas_pos[index+1] = shoulder_pos
                shoulder_meas_vel[index+1] = shoulder_vel
                shoulder_meas_tau[index] = tau_cmd[0]
                elbow_meas_pos[index+1] = elbow_pos
                elbow_meas_vel[index+1] = elbow_vel
                elbow_meas_tau[index] = tau_cmd[1]
                shoulder_filtered_meas_vel[index+1] = shoulder_filtered_vel
                elbow_filtered_meas_vel[index+1] = elbow_filtered_vel

                # wait to enforce the demanded control frequency
                meas_dt = time.time() - start_loop
                if meas_dt > dt:
                    print("Control loop is too slow!")
                    print("Control frequency:", 1/meas_dt, "Hz")
                    print("Desired frequency:", 1/dt, "Hz")
                    print()
                while time.time() - start_loop < dt:
                    pass

                # store times
                meas_time[index] = t
                index += 1
                t += time.time() - start_loop

        #     print("Disabling Motors...")
        #     (shoulder_pos,
        #      shoulder_vel,
        #      shoulder_tau) = motor_shoulder_controller.disable_motor()
        #     print(
        #        "Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
        #         shoulder_pos, shoulder_vel, shoulder_tau))
        #     (elbow_pos,
        #      elbow_vel,
        #      elbow_tau) = motor_elbow_controller.disable_motor()
        #     print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
        #         elbow_pos, elbow_vel, elbow_tau))

        # except BaseException:
        #     print('*******Exception Block!********')
        #     date = datetime.now().strftime("%Y%m%d-%I%M%S-%p")
        #     save_dir_time = os.path.join(save_dir, date)
        #     if not os.path.exists(save_dir_time):
        #         os.makedirs(save_dir_time)
        #     save_data(save_dir_time,
        #               date,
        #               shoulder_meas_pos,
        #               shoulder_meas_vel,
        #               shoulder_meas_tau,
        #               elbow_meas_pos,
        #               elbow_meas_vel,
        #               elbow_meas_tau,
        #               meas_time,
        #               shoulder_on)
        #     date = plot_figure(save_dir_time,
        #                        date,
        #                        shoulder_meas_pos,
        #                        shoulder_meas_vel,
        #                        shoulder_meas_tau,
        #                        elbow_meas_pos,
        #                        elbow_meas_vel,
        #                        elbow_meas_tau,
        #                        meas_time,
        #                        shoulder_on)
        #     plt.figure()
        #     plt.plot(meas_time, shoulder_meas_vel)
        #     plt.plot(meas_time, shoulder_filtered_meas_vel)
        #     plt.plot(meas_time, elbow_meas_vel)
        #     plt.plot(meas_time, elbow_filtered_meas_vel)
        #     plt.legend(["shoulder meas vel",
        #                 "shoulder vel filtered",
        #                 "elbow meas vel",
        #                 "elbow vel filtered"])
        #     plt.show()

        finally:
            print("Disabling Motors...")

            (shoulder_pos,
             shoulder_vel,
             shoulder_tau) = motor_shoulder_controller.disable_motor()

            print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                shoulder_pos, shoulder_vel, shoulder_tau))

            (elbow_pos,
             elbow_vel,
             elbow_tau) = motor_elbow_controller.disable_motor()

            print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                elbow_pos, elbow_vel, elbow_tau))

            date = datetime.now().strftime("%Y%m%d-%I%M%S-%p")
            save_dir_time = os.path.join(save_dir, date)
            if not os.path.exists(save_dir_time):
                os.makedirs(save_dir_time)
            save_data(save_dir_time,
                      date,
                      shoulder_meas_pos,
                      shoulder_meas_vel,
                      shoulder_meas_tau,
                      elbow_meas_pos,
                      elbow_meas_vel,
                      elbow_meas_tau,
                      meas_time,
                      shoulder_on)
            date = plot_figure(save_dir_time,
                               date,
                               shoulder_meas_pos,
                               shoulder_meas_vel,
                               shoulder_meas_tau,
                               elbow_meas_pos,
                               elbow_meas_vel,
                               elbow_meas_tau,
                               meas_time,
                               shoulder_on)
            plt.figure()
            plt.plot(meas_time, shoulder_meas_vel)
            plt.plot(meas_time, shoulder_filtered_meas_vel)
            plt.plot(meas_time, elbow_meas_vel)
            plt.plot(meas_time, elbow_filtered_meas_vel)
            plt.legend(["shoulder meas vel",
                        "shoulder vel filtered",
                        "elbow meas vel",
                        "elbow vel filtered"])
            plt.show()
