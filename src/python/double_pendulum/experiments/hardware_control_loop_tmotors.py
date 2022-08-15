import os
import time
from datetime import datetime
import numpy as np

from motor_driver.canmotorlib import CanMotorController
from double_pendulum.experiments.experimental_utils import setZeroPosition
from double_pendulum.utils.csv_trajectory import save_trajectory
from double_pendulum.utils.plotting import plot_timeseries, plot_figures


def run_experiment(controller,
                   dt=0.01,
                   t_final=10,
                   can_port='can0',
                   motor_ids=[8, 9],
                   motor_type='AK80_6_V1p1',
                   tau_limit=[4., 4.],
                   save_dir="."):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    n = int(t_final/dt) + 2

    meas_time = np.zeros(n+1)
    pos_meas1 = np.zeros(n+1)
    vel_meas1 = np.zeros(n+1)
    tau_meas1 = np.zeros(n+1)
    pos_meas2 = np.zeros(n+1)
    vel_meas2 = np.zeros(n+1)
    tau_meas2 = np.zeros(n+1)

    print("Enabling Motors..")
    motor_shoulder_id = motor_ids[0]
    motor_elbow_id = motor_ids[1]

    # Create motor controller objects
    motor_shoulder_controller = CanMotorController(can_port, motor_shoulder_id, motor_type)
    motor_elbow_controller = CanMotorController(can_port, motor_elbow_id, motor_type)

    (shoulder_pos,
     shoulder_vel,
     shoulder_torque) = motor_shoulder_controller.send_rad_command(
        0.0, 0.0, 0.0, 0.0, 0.0)

    (elbow_pos,
     elbow_vel,
     elbow_torque) = motor_elbow_controller.send_rad_command(
        0.0, 0.0, 0.0, 0.0, 0.0)

    print("Setting Shoulder Motor to Zero Position...")
    setZeroPosition(motor_shoulder_controller, shoulder_pos, shoulder_vel, shoulder_torque)

    print("Setting Elbow Motor to Zero Position...")
    setZeroPosition(motor_elbow_controller, elbow_pos, elbow_vel, elbow_torque)

    (shoulder_pos,
     shoulder_vel,
     shoulder_torque) = motor_shoulder_controller.enable_motor()
    print("Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
        shoulder_pos, shoulder_vel, shoulder_torque))

    elbow_pos, elbow_vel, elbow_torque = motor_elbow_controller.enable_motor()
    print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
        elbow_pos, elbow_vel, elbow_torque))

    if input('Do you want to proceed for real time execution?(y) ') == 'y':

        (shoulder_pos,
         shoulder_vel,
         shoulder_tau) = motor_shoulder_controller.send_rad_command(
            0.0, 0.0, 0.0, 0.0, 0.0)

        (elbow_pos,
         elbow_vel,
         elbow_tau) = motor_elbow_controller.send_rad_command(
            0.0, 0.0, 0.0, 0.0, 0.0)

        # defining running index variables
        index = 0
        t = 0.

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

                x = np.array([shoulder_pos,
                              elbow_pos,
                              shoulder_vel,
                              elbow_vel])

                # get control command from controller
                tau_cmd = controller.get_control_output(x, t)

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
                    0.0, 0.0, 0.0, 0.0, tau_cmd[1])

                # store the measured sensor data of
                # position, velocity and torque in each time step
                pos_meas1[index] = shoulder_pos
                vel_meas1[index] = shoulder_vel
                tau_meas1[index] = shoulder_tau
                pos_meas2[index] = elbow_pos
                vel_meas2[index] = elbow_vel
                tau_meas2[index] = elbow_tau

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
                t += time.time() - start_loop
                meas_time[index] = t

                index += 1
                # end of control loop

            try:
                print("Disabling Motors...")
                (shoulder_pos,
                 shoulder_vel,
                 shoulder_tau) = motor_shoulder_controller.disable_motor()
                print(
                   "Shoulder Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                    shoulder_pos, shoulder_vel, shoulder_tau))
                (elbow_pos,
                 elbow_vel,
                 elbow_tau) = motor_elbow_controller.disable_motor()
                print("Elbow Motor Status: Pos: {}, Vel: {}, Torque: {}".format(
                    elbow_pos, elbow_vel, elbow_tau))
            except TypeError:
                pass

        except BaseException as e:
            print('*******Exception Block!********')
            print(e)

        finally:
            #if motors_enabled:
            try:
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
            except TypeError:
                pass

            date = datetime.now().strftime("%Y%m%d-%I%M%S-%p")
            save_dir_time = os.path.join(save_dir, date)
            if not os.path.exists(save_dir_time):
                os.makedirs(save_dir_time)

            X_meas = np.asarray([pos_meas1[:index-1],
                                 pos_meas2[:index-1],
                                 vel_meas1[:index-1],
                                 vel_meas2[:index-1]]).T

            U_meas = np.asarray([tau_meas1[:index-1], tau_meas2[:index-1]]).T

            T_des, X_des, U_des = controller.get_init_trajectory()
            if len(T_des) <= 0:
                T_des = None

            if len(X_des) > 0:
                shoulder_des_pos = X_des.T[0]
                shoulder_des_vel = X_des.T[2]
                elbow_des_pos = X_des.T[1]
                elbow_des_vel = X_des.T[3]
            else:
                shoulder_des_pos = None
                shoulder_des_vel = None
                elbow_des_pos = None
                elbow_des_vel = None
                X_des = None

            if len(U_des) > 0:
                shoulder_des_tau = U_des.T[0]
                elbow_des_tau = U_des.T[1]
            else:
                shoulder_des_tau = None
                elbow_des_tau = None
                U_des = None

            save_trajectory(os.path.join(save_dir_time, "trajectory.csv"),
                            T=meas_time[:index-1],
                            X=None,
                            U=None,
                            ACC=None,
                            X_meas=X_meas,
                            X_filt=np.asarray(controller.x_filt_hist),
                            X_des=X_des,
                            U_con=np.asarray(controller.u_hist),
                            U_fric=np.asarray(controller.u_fric_hist),
                            U_meas=U_meas,
                            U_des=U_des,
                            K=None,
                            k=None)

            plot_timeseries(T=meas_time[:index-1],
                            X=X_meas,
                            U=U_meas,
                            T_des=T_des,
                            X_des=X_des,
                            U_des=U_des,
                            X_filt=np.asarray(controller.x_filt_hist)[:index-1],
                            U_con=np.asarray(controller.u_hist)[:index-1],
                            U_friccomp=np.asarray(controller.u_fric_hist)[:index-1],
                            save_to=os.path.join(save_dir_time, "combiplot.pdf"),
                            show=True)
                            
            plot_figures(save_dir=save_dir_time,
                         index=index-1,
                         meas_time=meas_time,
                         shoulder_meas_pos=pos_meas1,
                         shoulder_meas_vel=vel_meas1,
                         shoulder_meas_tau=tau_meas1,
                         elbow_meas_pos=pos_meas2,
                         elbow_meas_vel=vel_meas2,
                         elbow_meas_tau=tau_meas2,
                         shoulder_tau_controller=np.asarray(controller.u_hist).T[0],
                         elbow_tau_controller=np.asarray(controller.u_hist).T[1],
                         shoulder_des_time=T_des,
                         shoulder_des_pos=shoulder_des_pos,
                         shoulder_des_vel=shoulder_des_vel,
                         shoulder_des_tau=shoulder_des_tau,
                         elbow_des_time=T_des,
                         elbow_des_pos=elbow_des_pos,
                         elbow_des_vel=elbow_des_vel,
                         elbow_des_tau=elbow_des_tau,
                         error=None)

            # plot_figure_single(save_dir=save_dir_time,
            #                    date=date,
            #                    index=index-1,
            #                    meas_time=meas_time,
            #                    shoulder_meas_pos=pos_meas1,
            #                    shoulder_meas_vel=vel_meas1,
            #                    shoulder_meas_tau=tau_meas1,
            #                    elbow_meas_pos=pos_meas2,
            #                    elbow_meas_vel=vel_meas2,
            #                    elbow_meas_tau=tau_meas2,
            #                    shoulder_tau_controller=np.asarray(controller.u_hist).T[1],
            #                    elbow_tau_controller=np.asarray(controller.u_hist).T[0],
            #                    shoulder_des_time=T_des,
            #                    shoulder_des_pos=shoulder_des_pos,
            #                    shoulder_des_vel=shoulder_des_vel,
            #                    shoulder_des_tau=shoulder_des_tau,
            #                    elbow_des_time=T_des,
            #                    elbow_des_pos=elbow_des_pos,
            #                    elbow_des_vel=elbow_des_vel,
            #                    elbow_des_tau=elbow_des_tau,
            #                    error=None)
