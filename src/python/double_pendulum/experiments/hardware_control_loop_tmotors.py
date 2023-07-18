import os
import time
from datetime import datetime
import numpy as np

from motor_driver.canmotorlib import CanMotorController
from double_pendulum.experiments.experimental_utils import setZeroPosition
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries, plot_figures
from double_pendulum.experiments.video_recording import VideoWriterWidget

def run_experiment(
    controller,
    dt=0.01,
    t_final=10.0,
    can_port="can0",
    motor_ids=[1, 2],
    motor_type="AK80_6_V1p1",
    tau_limit=[6.0, 6.0],
    save_dir=".",
    record_video=True,
):
    """run_experiment.
    Hardware control loop for tmotor system.

    Parameters
    ----------
    controller : controller object
        controller which gives the control signal
    dt : float
        timestep of the control, unit=[s]
        (Default value=0.01)
    t_final : float
        duration of the experiment
        (Default value=10.)
    can_port : string
        the can port which is used to control the motors
        (Default value="can0")
    motor_ids : list
        shape=(2,), dtype=int
        ids of the 2 motors
        (Default value=[8, 9])
    motor_type : string
        the motor type being used
        (Default value="AK80_6_V1p1")
    tau_limit : array_like, optional
        shape=(2,), dtype=float,
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
        (Default value=[4., 4.])
    save_dir : string of path object
        directory where log data will be stored
        (Default value=".")
    """

    np.set_printoptions(formatter={"float": lambda x: "{0:0.4f}".format(x)})

    safety_position_limit = 2 * np.pi
    safety_velocity_limit = 20.0

    n = int(t_final / dt) + 2

    meas_time = np.zeros(n + 1)
    pos_meas1 = np.zeros(n + 1)
    vel_meas1 = np.zeros(n + 1)
    tau_meas1 = np.zeros(n + 1)
    pos_meas2 = np.zeros(n + 1)
    vel_meas2 = np.zeros(n + 1)
    tau_meas2 = np.zeros(n + 1)

    print("Enabling Motors..")
    motor_shoulder_id = motor_ids[0]
    motor_elbow_id = motor_ids[1]

    # Create motor controller objects
    motor_shoulder_controller = CanMotorController(
        can_port, motor_shoulder_id, motor_type
    )
    motor_elbow_controller = CanMotorController(can_port, motor_elbow_id, motor_type)

    (
        shoulder_pos,
        shoulder_vel,
        shoulder_torque,
    ) = motor_shoulder_controller.enable_motor()
    print(
        "Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
            shoulder_pos, shoulder_vel, shoulder_torque
        )
    )

    elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()
    print(
        "Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
            elbow_pos, elbow_vel, elbow_torque
        )
    )

    (
        shoulder_pos,
        shoulder_vel,
        shoulder_torque,
    ) = motor_shoulder_controller.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)

    (elbow_pos, elbow_vel, elbow_torque) = motor_elbow_controller.send_rad_command(
        0.0, 0.0, 0.0, 0.0, 0.0
    )

    print("Setting Shoulder Motor to Zero Position...")
    setZeroPosition(
        motor_shoulder_controller, shoulder_pos, shoulder_vel, shoulder_torque
    )

    print("Setting Elbow Motor to Zero Position...")
    setZeroPosition(motor_elbow_controller, elbow_pos, elbow_vel, elbow_torque)

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir_time = os.path.join(save_dir, date)
    if not os.path.exists(save_dir_time):
        os.makedirs(save_dir_time)

    if input("Do you want to proceed for real time execution? (y/N) ") == "y":
        (
            shoulder_pos,
            shoulder_vel,
            shoulder_tau,
        ) = motor_shoulder_controller.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)

        (elbow_pos, elbow_vel, elbow_tau) = motor_elbow_controller.send_rad_command(
            0.0, 0.0, 0.0, 0.0, 0.0
        )

        # video recorder
        if record_video:
            video_writer = VideoWriterWidget(os.path.join(save_dir_time, "video"), 0)

        last_shoulder_pos = shoulder_pos
        last_elbow_pos = elbow_pos

        # defining running index variables
        index = 0
        t = 0.0

        meas_time[0] = t
        pos_meas1[0] = shoulder_pos
        vel_meas1[0] = shoulder_vel
        pos_meas2[0] = elbow_pos
        vel_meas2[0] = elbow_vel
        tau_meas1[0] = shoulder_tau
        tau_meas2[0] = elbow_tau

        index += 1

        print("Starting Experiment...")
        # start_time = time.time()
        try:
            while t < t_final:
                start_loop = time.time()

                x = np.array([shoulder_pos, elbow_pos, shoulder_vel, elbow_vel])

                # get control command from controller
                tau_cmd = controller.get_control_output(x, t)

                # safety command
                tau_cmd[0] = np.clip(tau_cmd[0], -tau_limit[0], tau_limit[0])
                tau_cmd[1] = np.clip(tau_cmd[1], -tau_limit[1], tau_limit[1])

                # Send tau command to motors
                (
                    shoulder_pos,
                    shoulder_vel,
                    shoulder_tau,
                ) = motor_shoulder_controller.send_rad_command(
                    0.0, 0.0, 0.0, 0.0, tau_cmd[0]
                )

                (
                    elbow_pos,
                    elbow_vel,
                    elbow_tau,
                ) = motor_elbow_controller.send_rad_command(
                    0.0, 0.0, 0.0, 0.0, tau_cmd[1]
                )

                # store the measured sensor data of
                # position, velocity and torque in each time step
                pos_meas1[index] = shoulder_pos
                #vel_meas1[index] = shoulder_vel
                tau_meas1[index] = shoulder_tau
                pos_meas2[index] = elbow_pos
                #vel_meas2[index] = elbow_vel
                tau_meas2[index] = elbow_tau

                # wait to enforce the demanded control frequency
                meas_dt = time.time() - start_loop
                if meas_dt > dt:
                    print("Control loop is too slow!")
                    print("Control frequency:", 1 / meas_dt, "Hz")
                    print("Desired frequency:", 1 / dt, "Hz")
                    print()
                while time.time() - start_loop < dt:
                    pass

                # store times
                t += time.time() - start_loop
                meas_time[index] = t

                # velocities from position measurements
                shoulder_vel = (shoulder_pos - last_shoulder_pos) / (
                    meas_time[index] - meas_time[index - 1]
                )
                elbow_vel = (elbow_pos - last_elbow_pos) / (
                    meas_time[index] - meas_time[index - 1]
                )
                last_shoulder_pos = shoulder_pos
                last_elbow_pos = elbow_pos
                vel_meas1[index] = shoulder_vel
                vel_meas2[index] = elbow_vel

                index += 1
                # end of control loop

                # check safety conditions
                if (
                    np.abs(shoulder_pos) > safety_position_limit
                    or np.abs(elbow_pos) > safety_position_limit
                    or np.abs(shoulder_vel) > safety_velocity_limit
                    or np.abs(elbow_vel) > safety_velocity_limit
                ):
                    for _ in range(int(1 / dt)):
                        # send kd command to slow down motors for 1 second
                        _ = motor_elbow_controller.send_rad_command(
                            0.0, 0.0, 0.0, 1.0, 0.0
                        )
                        _ = motor_shoulder_controller.send_rad_command(
                            0.0, 0.0, 0.0, 1.0, 0.0
                        )
                    print("Safety conditions violated! Stopping experiment.")
                    print("The measured state violating the limits was ({}, {}, {}, {})".format(
                          shoulder_pos,
                          elbow_pos,
                          shoulder_vel,
                          elbow_vel))
                    break

            try:
                print("Disabling Motors...")
                (
                    shoulder_pos,
                    shoulder_vel,
                    shoulder_tau,
                ) = motor_shoulder_controller.disable_motor()
                print(
                    "Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                        shoulder_pos, shoulder_vel, shoulder_tau
                    )
                )
                (
                    elbow_pos,
                    elbow_vel,
                    elbow_tau,
                ) = motor_elbow_controller.disable_motor()
                print(
                    "Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                        elbow_pos, elbow_vel, elbow_tau
                    )
                )
            except TypeError:
                pass

        except BaseException as e:
            print("*******Exception Block!********")
            print(e)

        finally:
            # if motors_enabled:
            try:
                print("Disabling Motors...")

                (
                    shoulder_pos,
                    shoulder_vel,
                    shoulder_tau,
                ) = motor_shoulder_controller.disable_motor()

                print(
                    "Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                        shoulder_pos, shoulder_vel, shoulder_tau
                    )
                )

                (
                    elbow_pos,
                    elbow_vel,
                    elbow_tau,
                ) = motor_elbow_controller.disable_motor()

                print(
                    "Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                        elbow_pos, elbow_vel, elbow_tau
                    )
                )

                # stop video recording
            except TypeError:
                pass

            X_meas = np.asarray(
                [
                    pos_meas1[: index - 1],
                    pos_meas2[: index - 1],
                    vel_meas1[: index - 1],
                    vel_meas2[: index - 1],
                ]
            ).T

            U_meas = np.asarray([tau_meas1[: index - 1], tau_meas2[: index - 1]]).T

            T_des, X_des, U_des = controller.get_init_trajectory()
            if len(T_des) <= 0:
                T_des = None

            if len(X_des) > 0:
                shoulder_des_pos = np.asarray(X_des).T[0]
                shoulder_des_vel = np.asarray(X_des).T[2]
                elbow_des_pos = np.asarray(X_des).T[1]
                elbow_des_vel = np.asarray(X_des).T[3]

            else:
                shoulder_des_pos = None
                shoulder_des_vel = None
                elbow_des_pos = None
                elbow_des_vel = None
                X_des = None

            if len(U_des) > 0:
                shoulder_des_tau = np.asarray(U_des).T[0]
                elbow_des_tau = np.asarray(U_des).T[1]
            else:
                shoulder_des_tau = None
                elbow_des_tau = None
                U_des = None

            save_trajectory(
                os.path.join(save_dir_time, "trajectory.csv"),
                T=meas_time[: index - 1],
                X=None,
                U=None,
                ACC=None,
                X_meas=X_meas,
                X_filt=np.asarray(controller.x_filt_hist),
                X_des=X_des,
                U_con=np.asarray(controller.u_hist[1:]),
                U_fric=np.asarray(controller.u_fric_hist),
                U_meas=U_meas,
                U_des=U_des,
                K=None,
                k=None,
            )

            plot_timeseries(
                T=meas_time[: index - 1],
                X=X_meas,
                U=U_meas,
                T_des=T_des,
                X_des=X_des,
                U_des=U_des,
                X_filt=np.asarray(controller.x_filt_hist)[: index - 1],
                U_con=np.asarray(controller.u_hist)[1 : index - 1],
                U_friccomp=np.asarray(controller.u_fric_hist)[: index - 1],
                pos_y_lines=[-safety_position_limit, -np.pi, 0.0, np.pi, safety_position_limit],
                vel_y_lines=[-safety_velocity_limit, 0.0, safety_velocity_limit],
                tau_y_lines=[-tau_limit[0], -tau_limit[1], 0.0, tau_limit[0], tau_limit[1]],
                save_to=os.path.join(save_dir_time, "timeseries"),
                show=True,
            )

            plot_figures(
                save_dir=save_dir_time,
                index=index - 1,
                meas_time=meas_time,
                shoulder_meas_pos=pos_meas1,
                shoulder_meas_vel=vel_meas1,
                shoulder_meas_tau=tau_meas1,
                elbow_meas_pos=pos_meas2,
                elbow_meas_vel=vel_meas2,
                elbow_meas_tau=tau_meas2,
                shoulder_tau_controller=np.asarray(controller.u_hist[1:]).T[0],
                elbow_tau_controller=np.asarray(controller.u_hist[1:]).T[1],
                shoulder_des_time=T_des,
                shoulder_des_pos=shoulder_des_pos,
                shoulder_des_vel=shoulder_des_vel,
                shoulder_des_tau=shoulder_des_tau,
                elbow_des_time=T_des,
                elbow_des_pos=elbow_des_pos,
                elbow_des_vel=elbow_des_vel,
                elbow_des_tau=elbow_des_tau,
                error=None,
                show=False,
            )
    else:
        print("Disabling Motors...")

        (
            shoulder_pos,
            shoulder_vel,
            shoulder_tau,
        ) = motor_shoulder_controller.disable_motor()

        print(
            "Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                shoulder_pos, shoulder_vel, shoulder_tau
            )
        )

        (elbow_pos, elbow_vel, elbow_tau) = motor_elbow_controller.disable_motor()

        print(
            "Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                elbow_pos, elbow_vel, elbow_tau
            )
        )
