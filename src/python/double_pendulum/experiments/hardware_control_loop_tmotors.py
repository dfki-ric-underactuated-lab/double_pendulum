import os
import time
from datetime import datetime
import numpy as np

from motor_driver.canmotorlib import CanMotorController
from double_pendulum.experiments.experimental_utils import (
    setZeroPosition,
    enable_motor,
    disable_motor,
    go_to_zero,
)
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries
from double_pendulum.experiments.video_recording import VideoWriterWidget
from double_pendulum.simulation.perturbations import plot_perturbation_array


def run_experiment(
    controller,
    dt=0.01,
    t_final=10.0,
    can_port="can0",
    motor_ids=[1, 2],
    motor_directions=[1.0, 1.0],
    motor_type="AK80_6_V1p1",
    tau_limit=[6.0, 6.0],
    save_dir=".",
    record_video=False,
    safety_position_limit=4.0 * np.pi,
    safety_velocity_limit=20.0,
    perturbation_array=None,
):
    """run_experiment.
    Hardware control loop for tmotor system.

    Parameters
    ----------
    controller : controller object
        controller which gives the control signal
    dt : float, optional
        timestep of the control, unit=[s]
        (Default value=0.01)
    t_final : float, optional
        duration of the experiment
        (Default value=10.)
    can_port : string, optional
        the can port which is used to control the motors
        (Default value="can0")
    motor_ids : list, optional
        shape=(2,), dtype=int
        ids of the 2 motors
        (Default value=[8, 9])
    motor_type : string, optional
        the motor type being used
        (Default value="AK80_6_V1p1")
    tau_limit : array_like, optional
        shape=(2,), dtype=float,
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
        (Default value=[4., 4.])
    save_dir : string of path object, optional
        directory where log data will be stored
        (Default value=".")
    record_video : bool, optional
        whether to rocird a video with a camera during the execution
        (Default value=False)
    safety_position_limit : float, optional
        safety limit for motor positions. Execution will stop if the limits
        are violated
        (Default value=2.0*np.pi)
    safety_velocity_limit : float, optional
        safety limit for motor velocities. Execution will stop if the limits
        are violated
        (Default value=20.0)
    perturbation_array : np.array, optional
        shape=(2, N), with N = t_final/dt,
        perturbations to be applied during the execution.
        If set to None, no perturbations are applied.
        Perturbations can exceed the tau_limits!
        (Default value=None)
    """

    np.set_printoptions(formatter={"float": lambda x: "{0:0.4f}".format(x)})

    if perturbation_array is None:
        perturbation_array = np.zeros((2, int(t_final / dt)))

    n = int(t_final / dt)

    meas_time = np.zeros(n + 2)
    pos_meas1 = np.zeros(n + 2)
    vel_meas1 = np.zeros(n + 2)
    tau_meas1 = np.zeros(n + 2)
    pos_meas2 = np.zeros(n + 2)
    vel_meas2 = np.zeros(n + 2)
    tau_meas2 = np.zeros(n + 2)

    tau_cmd = np.zeros(2)

    # Create motor controller objects
    motor1 = CanMotorController(can_port, motor_ids[0], motor_type)
    motor2 = CanMotorController(can_port, motor_ids[1], motor_type)

    enable_motor(motor1)
    enable_motor(motor2)

    setZeroPosition(motor1, motor_directions[0])
    setZeroPosition(motor2, motor_directions[1])

    if input("Do you want to proceed for real time execution? (y/N) ") != "y":
        disable_motor(motor1)
        disable_motor(motor2)
    else:
        # create save directory
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir_time = os.path.join(save_dir, date)
        if not os.path.exists(save_dir_time):
            os.makedirs(save_dir_time)

        # video recorder
        if record_video:
            video_writer = VideoWriterWidget(os.path.join(save_dir_time, "video"), 0)

        # initial measurements
        pos1, vel1, tau1 = motor1.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
        pos2, vel2, tau2 = motor2.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)

        # correction for motor axis directions
        pos1 *= motor_directions[0]
        vel1 *= motor_directions[0]
        tau1 *= motor_directions[0]
        pos2 *= motor_directions[1]
        vel2 *= motor_directions[1]
        tau2 *= motor_directions[1]

        last_pos1 = pos1
        last_pos2 = pos2

        # defining running index variables
        index = 0
        index_t = 0
        t = 0.0

        meas_time[0] = t
        pos_meas1[0] = pos1
        vel_meas1[0] = vel1
        tau_meas1[0] = tau1
        pos_meas2[0] = pos2
        vel_meas2[0] = vel2
        tau_meas2[0] = tau2

        index += 1

        time.sleep(1.0)

        print("Starting Experiment...")
        try:
            while t < t_final:
                start_loop = time.time()

                x = np.array([pos1, pos2, vel1, vel2])

                # get control command from controller
                tau_con = controller.get_control_output(x, t)

                # safety command
                tau_cmd[0] = np.clip(tau_con[0], -tau_limit[0], tau_limit[0])
                tau_cmd[1] = np.clip(tau_con[1], -tau_limit[1], tau_limit[1])

                # perturbations
                index_t = int(t / dt)
                tau_cmd[0] += perturbation_array[0, min(index_t, n - 1)]
                tau_cmd[1] += perturbation_array[1, min(index_t, n - 1)]

                # play terminal bell sound when perturbation is active
                if np.max(np.abs(perturbation_array[:, min(index, n - 1)])) > 0.1:
                    print("\a", end="\r")

                # correction for motor axis directions
                tau_cmd[0] *= motor_directions[0]
                tau_cmd[1] *= motor_directions[1]

                # Send tau command to motors
                (
                    pos1,
                    vel1,
                    tau1,
                ) = motor1.send_rad_command(0.0, 0.0, 0.0, 0.0, tau_cmd[0])

                (
                    pos2,
                    vel2,
                    tau2,
                ) = motor2.send_rad_command(0.0, 0.0, 0.0, 0.0, tau_cmd[1])

                # correction for motor axis directions
                pos1 *= motor_directions[0]
                vel1 *= motor_directions[0]
                tau1 *= motor_directions[0]
                pos2 *= motor_directions[1]
                vel2 *= motor_directions[1]
                tau2 *= motor_directions[1]

                # store the measured sensor data of
                # position, velocity and torque in each time step
                pos_meas1[index] = pos1
                # vel_meas1[index] = vel1
                tau_meas1[index] = tau1
                pos_meas2[index] = pos2
                # vel_meas2[index] = vel2
                tau_meas2[index] = tau2

                # check control loop speed
                meas_dt = time.time() - start_loop
                if meas_dt > dt:
                    print(f"[{t}] Control loop too slow!", end=" ")
                    print(f"Desired: {1 / dt}Hz, Actual: {1 / meas_dt}Hz")

                # wait to enforce the demanded control frequency
                while time.time() - start_loop < dt:
                    pass

                # store times
                t += time.time() - start_loop
                meas_time[index] = t

                # velocities from position measurements
                vel1 = (pos1 - last_pos1) / (meas_time[index] - meas_time[index - 1])
                vel2 = (pos2 - last_pos2) / (meas_time[index] - meas_time[index - 1])
                last_pos1 = pos1
                last_pos2 = pos2
                vel_meas1[index] = vel1
                vel_meas2[index] = vel2

                index += 1
                # end of control loop

                # check safety conditions
                if (
                    np.abs(pos1) > safety_position_limit
                    or np.abs(pos2) > safety_position_limit
                    or np.abs(vel1) > safety_velocity_limit
                    or np.abs(vel2) > safety_velocity_limit
                ):
                    for _ in range(int(1 / dt)):
                        # send kd command to slow down motors for 1 second
                        _ = motor1.send_rad_command(0.0, 0.0, 0.0, 1.0, 0.0)
                        _ = motor2.send_rad_command(0.0, 0.0, 0.0, 1.0, 0.0)
                    print("Safety conditions violated! Stopping experiment.")
                    print(
                        f"The limit violating state was ({pos1}, {pos2}, {vel1}, {vel2})"
                    )
                    break

        except BaseException as e:
            print("*******Exception Block!********")
            print(e)

        finally:
            try:
                _ = motor1.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
                _ = motor2.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
            except TypeError:
                pass
            if record_video:
                video_writer.stop_threads()
            if input("Reset the pendulum to initial configuration? (y/N) ") == "y":
                go_to_zero(motor1, motor2, motor_directions)
            disable_motor(motor1)
            disable_motor(motor2)

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

            save_trajectory(
                os.path.join(save_dir_time, "trajectory.csv"),
                T=meas_time[: index - 1],
                X=None,
                U=None,
                ACC=None,
                X_meas=X_meas,
                X_filt=np.asarray(controller.filter.x_hist),
                X_des=X_des,
                U_con=np.asarray(controller.u_hist[1:]),
                U_fric=np.asarray(controller.u_fric_hist),
                U_meas=U_meas,
                U_des=U_des,
                U_perturbation=perturbation_array.T,
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
                U_perturbation=perturbation_array.T,
                X_filt=np.asarray(controller.filter.x_hist)[: index - 1],
                U_con=np.asarray(controller.u_hist)[1 : index - 1],
                U_friccomp=np.asarray(controller.u_fric_hist)[: index - 1],
                pos_y_lines=[
                    -safety_position_limit,
                    -np.pi,
                    0.0,
                    np.pi,
                    safety_position_limit,
                ],
                vel_y_lines=[-safety_velocity_limit, 0.0, safety_velocity_limit],
                tau_y_lines=[
                    -tau_limit[0],
                    -tau_limit[1],
                    0.0,
                    tau_limit[0],
                    tau_limit[1],
                ],
                save_to=os.path.join(save_dir_time, "timeseries"),
                show=True,
            )
            plot_perturbation_array(
                t_final,
                dt,
                perturbation_array,
                save_to=os.path.join(save_dir_time, "perturbations"),
                show=True,
            )
