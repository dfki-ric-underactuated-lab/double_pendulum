import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from double_pendulum.experiments.canmotorlib import CanMotorController
from double_pendulum.experiments.experimental_utils import (yb_friction_matrix,
                                                           read_data,
                                                           wrap_angle_pi2pi,
                                                           prepare_data,
                                                           prepare_empty_data,
                                                           plot_figure,
                                                           save_data,
                                                           setZeroPosition)

from double_pendulum.experiments.filters.low_pass import online_filter

def run_experiment(controller,
                   dt=0.01,
                   t_final=10,
                   can_port='can0',
                   tau_limit=[4., 4.],
                   friction_compensation=False,
                   Fc1=0.093,
                   Fc2=0.186,
                   Fv1=0.081,
                   Fv2=0.0,
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
    motor_shoulder_id = 0x08
    motor_elbow_id = 0x09

    # Create motor controller objects
    motor_shoulder_controller = CanMotorController(can_port, motor_shoulder_id)
    motor_elbow_controller = CanMotorController(can_port, motor_elbow_id)

    shoulder_pos, shoulder_vel, shoulder_torque = motor_shoulder_controller.enable_motor()
    print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_torque))

    elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()
    print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_torque))

    print("Setting Shoulder Motor to Zero Position...")
    setZeroPosition(motor_shoulder_controller, shoulder_pos)

    print("Setting Elbow Motor to Zero Position...")
    setZeroPosition(motor_elbow_controller, elbow_pos)

    shoulder_meas_pos[0] = shoulder_pos
    shoulder_meas_vel[0] = shoulder_vel
    elbow_meas_pos[0] = elbow_pos
    elbow_meas_vel[0] = elbow_vel

    if input('Do you want to proceed for real time execution?(y) ') == 'y':
        print(np.array([shoulder_pos,
                      elbow_pos,
                      shoulder_vel,
                      elbow_vel]))

        # if len(sys.argv) != 2:
        #     print('Provide CAN device name (can0, slcan0 etc.)')
        #     sys.exit(0)

        # print("Using Socket {} for can communucation".format(sys.argv[1],))

        # defining running index variables
        index = 0
        t = 0.
        meas_dt = 0.
        tau_fric = np.zeros(2)

        print("Starting Experiment...")
        start_time = time.time()
        try:
            # shoulder_pos = 0.
            # elbow_pos = 0.
            # shoulder_vel = 0.
            # elbow_vel = 0.
            while t < t_final:
                start_loop = time.time()
                # add the real_dt to the previous time step
                t += meas_dt

                n_filter = min(index+1, 20)

                shoulder_filtered_vel = online_filter(shoulder_meas_vel[max(index-n_filter, 0):index+1], n_filter, 0.3)
                elbow_filtered_vel = online_filter(elbow_meas_vel[max(index-n_filter, 0):index+1], n_filter, 0.3)

                shoulder_filtered_meas_vel[index] = shoulder_filtered_vel
                elbow_filtered_meas_vel[index] = elbow_filtered_vel

                x = np.array([shoulder_pos,
                              elbow_pos,
                              shoulder_filtered_vel,
                              elbow_filtered_vel])

                print(x)
                # get control command from controller
                tau_cmd = controller.get_control_output(x)

                # safety command
                tau_cmd[0] = np.clip(tau_cmd[0], -tau_limit[0], tau_limit[0])
                tau_cmd[1] = np.clip(tau_cmd[1], -tau_limit[1], tau_limit[1])

                # Send only the tau_ff command and use the in-built low level controller
                shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(0.0, 0.0, 0.0, 0.0, tau_cmd[0]+tau_fric[0])
                elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(0.0, 0.0, 0.0, 0.0, -tau_cmd[1]+tau_fric[1])

                #shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.send_rad_command(np.pi, 0.0, 0.0, 0.0, 0.0)
                #elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
                # friction compensation
                if friction_compensation:
                    friction_regressor_mat = yb_friction_matrix([shoulder_vel, elbow_vel])  # with measured vel
                    #friction_regressor_mat = yb_friction_matrix([tvlqr.x0.value(t)[2], tvlqr.x0.value(t)[3]])  # with desired vel
                    tau_fric = np.dot(friction_regressor_mat, np.array([Fc1, Fv1, Fc2, Fv2]))
                    tau_fric[0] = 0.0
                    #tau_fric[1] = 0.0

                # store the measured sensor data of position, velocity and torque in each time step
                shoulder_meas_pos[index+1] = shoulder_pos
                shoulder_meas_vel[index+1] = shoulder_vel
                shoulder_meas_tau[index] = tau_cmd[0]+tau_fric[0]
                elbow_meas_pos[index+1] = elbow_pos
                elbow_meas_vel[index+1] = elbow_vel
                elbow_meas_tau[index] = tau_cmd[1] + tau_fric[1]

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

            print("Disabling Motors...")
            shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()
            print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))
            elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()
            print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

        except BaseException as e:
            print('*******Exception Block!********')
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
            plt.legend(["shoulder meas vel", "shoulder vel filtered", "elbow meas vel", "elbow vel filtered"])
            plt.show()

        finally:
            print("Disabling Motors...")

            shoulder_pos, shoulder_vel, shoulder_tau = motor_shoulder_controller.disable_motor()

            print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(shoulder_pos, shoulder_vel, shoulder_tau))

            elbow_pos, elbow_vel, elbow_tau = motor_elbow_controller.disable_motor()

            print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(elbow_pos, elbow_vel, elbow_tau))

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
            plt.legend(["shoulder meas vel", "shoulder vel filtered", "elbow meas vel", "elbow vel filtered"])
            plt.show()
