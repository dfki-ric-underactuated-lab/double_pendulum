# import os
# import time
# from datetime import datetime
# import asyncio
# import moteus
# import numpy as np
# from scipy.signal import medfilt
# import matplotlib.pyplot as plt
#
# from double_pendulum.experiments.experimental_utils import (plot_figure,
#                                                             plot_figure_single,
#                                                             save_data,
#                                                             rev2rad,
#                                                             rad2rev)
#
# from double_pendulum.model.friction_matrix import yb_friction_matrix
# from double_pendulum.utils.filters.low_pass import lowpass_filter
#
#
# async def run_experiment(controller,
#                          dt=0.01,
#                          t_final=10,
#                          motor_ids=[1, 2],
#                          tau_limit=[4., 4.],
#                          friction_compensation=False,
#                          friction_terms=[0.093, 0.186, 0.081, 0.0],
#                          velocity_filter=None,
#                          filter_args={"alpha": 0.3,
#                                       "kernel_size": 5,
#                                       "filter_size": 1},
#                          save_dir="."):
#     """run_experiment.
#     Hardware control loop for mjbot system.
#     Has to be called with async.
#
#     Parameters
#     ----------
#     controller : controller object
#         controller which gives the control signal
#     dt : float
#         timestep of the control, unit=[s]
#         (Default value=0.01)
#     t_final : float
#         duration of the experiment
#         (Default value=10.)
#     motor_ids : list
#         shape=(2,), dtype=int
#         ids of the 2 motors
#         (Default value=[1, 2])
#     tau_limit : array_like, optional
#         shape=(2,), dtype=float,
#         torque limit of the motors
#         [tl1, tl2], units=[Nm, Nm]
#         (Default value=[4., 4.])
#     friction_compensation : bool
#         Whether to compensate for friction
#         Should be removed. Friction can be compensated for at controller level.
#         (Default value=False)
#     friction_terms : list
#         shape=(4,)
#         Friction terms used for the friction compensation.
#         Should be removed. Friction can be compensated for at controller level.
#         (Default value=[0.093, 0.186, 0.081, 0.0])
#     velocity_filter : string
#         string determining what velocity filter should be used.
#         Should be removed. Filtering can be done at controller level.
#         (Default value=None)
#     filter_args : dict
#         dictionary with arguments for the velocity filter
#         Should be removed. Filtering can be done at controller level.
#         (Default value={"alpha": 0.3, "kernel_size": 5, "filter_size": 1})
#     save_dir : string of path object
#         directory where log data will be stored
#         (Default value=".")
#     """
#
#     n = int(t_final/dt)
#
#     T_des, X_des, U_des = controller.get_init_trajectory()
#     if len(X_des) > 0:
#         shoulder_des_pos = X_des.T[0]
#         shoulder_des_vel = X_des.T[2]
#         elbow_des_pos = X_des.T[1]
#         elbow_des_vel = X_des.T[3]
#     else:
#         shoulder_des_pos = None
#         shoulder_des_vel = None
#         elbow_des_pos = None
#         elbow_des_vel = None
#
#     if len(U_des) > 0:
#         shoulder_des_tau = U_des.T[0]
#         elbow_des_tau = U_des.T[1]
#     else:
#         shoulder_des_tau = None
#         elbow_des_tau = None
#
#     # (shoulder_meas_pos,
#     #  shoulder_meas_vel,
#     #  shoulder_meas_tau,
#     #  elbow_meas_pos,
#     #  elbow_meas_vel,
#     #  elbow_meas_tau,
#     #  meas_time,
#     #  gear_ratio,
#     #  rad2outputrev,
#     #  shoulder_on) = prepare_empty_data(n+1)
#     meas_time = np.zeros(n+1)
#     shoulder_meas_pos = np.zeros(n+1)
#     shoulder_meas_vel = np.zeros(n+1)
#     shoulder_meas_tau = np.zeros(n)
#     elbow_meas_pos = np.zeros(n+1)
#     elbow_meas_vel = np.zeros(n+1)
#     elbow_meas_tau = np.zeros(n)
#     shoulder_tau_controller = np.zeros(n)
#     elbow_tau_controller = np.zeros(n)
#     shoulder_fric_tau = np.zeros(n)
#     elbow_fric_tau = np.zeros(n)
#     shoulder_filtered_meas_vel = np.zeros(n+1)
#     elbow_filtered_meas_vel = np.zeros(n+1)
#
#     tau_fric = np.zeros(2)
#
#     print("Enabling Motors..")
#     shoulder_id = motor_ids[0]
#     elbow_id = motor_ids[1]
#     sh_motor = moteus.Controller(id=shoulder_id)
#     el_motor = moteus.Controller(id=elbow_id)
#     # stop both motors
#     await sh_motor.set_stop()
#     await el_motor.set_stop()
#
#     hz = int(1 / dt)
#     os.system(f"sudo -E env PATH=$PATH moteus_tool --zero-offset -t{shoulder_id},{elbow_id}")
#
#     if input('Do you want to proceed for real time execution?(y) ') == 'y':
#         np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
#
#         # Read the intial values of the motor without sending command
#         sh_state = await sh_motor.set_position(position=np.nan, query=True)
#         shoulder_pos = rev2rad(sh_state.values[moteus.Register.POSITION])
#         shoulder_vel = rev2rad(sh_state.values[moteus.Register.VELOCITY])
#         el_state = await el_motor.set_position(position=np.nan, query=True)
#         elbow_pos = rev2rad(el_state.values[moteus.Register.POSITION])
#         elbow_vel = rev2rad(el_state.values[moteus.Register.VELOCITY])
#
#         meas_time[0] = 0.0
#         shoulder_meas_pos[0] = shoulder_pos
#         shoulder_meas_vel[0] = shoulder_vel
#         shoulder_filtered_meas_vel[0] = shoulder_vel
#         #shoulder_calc_meas_vel[0] = 0.0
#         elbow_meas_pos[0] = elbow_pos
#         elbow_meas_vel[0] = elbow_vel
#         elbow_filtered_meas_vel[0] = elbow_vel
#         #elbow_calc_meas_vel[0] = 0.0
#
#         last_shoulder_pos = shoulder_pos
#         last_elbow_pos = elbow_pos
#
#         shoulder_thresh_vel = 0.12
#         elbow_thresh_vel = 0.45
#         shoulder_vel = shoulder_vel if np.abs(shoulder_vel) > shoulder_thresh_vel else 0.0
#         elbow_vel = elbow_vel if np.abs(elbow_vel) > elbow_thresh_vel else 0.0
#
#         print(np.array([shoulder_pos,
#                         elbow_pos,
#                         shoulder_vel,
#                         elbow_vel]))
#
#         # defining running index variables
#         index = 0
#         t = 0.
#
#         print("Starting Experiment...")
#         # start_time = time.time()
#         try:
#             while t < t_final:
#                 start_loop = time.time()
#
#                 if velocity_filter == "lowpass":
#                     # the lowpass filter needs only the last filtered vel
#                     # and the latest vel value
#                     sfv = [shoulder_filtered_meas_vel[max(0, index-1)],
#                            shoulder_meas_vel[index]]
#                     efv = [elbow_filtered_meas_vel[max(0, index-1)],
#                            elbow_meas_vel[index]]
#                     shoulder_filtered_vel = lowpass_filter(
#                         sfv, filter_args["alpha"])[-1]
#                     elbow_filtered_vel = lowpass_filter(
#                         efv, filter_args["alpha"])[-1]
#                 elif velocity_filter == "medianfilter":
#                     n_filter = min(index+1, filter_args["filter_size"])
#                     shoulder_filtered_vel = medfilt(
#                         shoulder_meas_vel[max(index-n_filter, 0):index+1],
#                         kernel_size=filter_args["kernel_size"])[-1]
#                     elbow_filtered_vel = medfilt(
#                         elbow_meas_vel[max(index-n_filter, 0):index+1],
#                         kernel_size=filter_args["kernel_size"])[-1]
#                 elif velocity_filter == "meanfilter":
#                     n_filter = min(index+1, filter_args["filter_size"])
#                     shoulder_filtered_vel = np.mean(
#                         shoulder_meas_vel[max(index-n_filter, 0):index+1])
#                     elbow_filtered_vel = np.mean(
#                         elbow_meas_vel[max(index-n_filter, 0):index+1])
#                 else:
#                     shoulder_filtered_vel = shoulder_vel
#                     elbow_filtered_vel = elbow_vel
#
#                 x = np.array([shoulder_pos,
#                               elbow_pos,
#                               shoulder_filtered_vel,
#                               elbow_filtered_vel])
#
#                 # get control command from controller
#                 tau_cmd = controller.get_control_output(x)
#                 #tau_cmd = [0., 0.]
#
#                 # safety command
#                 tau_cmd[0] = np.clip(tau_cmd[0], -tau_limit[0], tau_limit[0])
#                 tau_cmd[1] = np.clip(tau_cmd[1], -tau_limit[1], tau_limit[1])
#
#                 shoulder_tau_controller[index] = tau_cmd[0]
#                 elbow_tau_controller[index] = tau_cmd[1]
#                 shoulder_fric_tau[index] = tau_fric[0]
#                 elbow_fric_tau[index] = tau_fric[1]
#
#                 # add friction compensation (0 if turned off)
#                 tau_cmd[0] += tau_fric[0]
#                 tau_cmd[1] += tau_fric[1]
#
#                 # Send tau command to motors
#                 sh_state = await sh_motor.set_position(position=0,
#                                                        velocity=0,
#                                                        kp_scale=0,
#                                                        kd_scale=0,
#                                                        stop_position=None,
#                                                        feedforward_torque=tau_cmd[0],
#                                                        maximum_torque=tau_limit[0],
#                                                        watchdog_timeout=None,
#                                                        query=True)
#                 el_state = await el_motor.set_position(position=0,
#                                                        velocity=0,
#                                                        kp_scale=0,
#                                                        kd_scale=0,
#                                                        stop_position=None,
#                                                        feedforward_torque=-tau_cmd[1],
#                                                        maximum_torque=tau_limit[1],
#                                                        watchdog_timeout=None,
#                                                        query=True)
#                 # sh_state = await sh_motor.set_position(position=np.nan, query=True)
#                 # el_state = await el_motor.set_position(position=np.nan, query=True)
#                 shoulder_pos = rev2rad(sh_state.values[moteus.Register.POSITION])
#                 shoulder_vel = rev2rad(sh_state.values[moteus.Register.VELOCITY])
#                 shoulder_tau = sh_state.values[moteus.Register.TORQUE]
#                 elbow_pos = rev2rad(el_state.values[moteus.Register.POSITION])
#                 elbow_vel = rev2rad(el_state.values[moteus.Register.VELOCITY])
#                 elbow_tau = el_state.values[moteus.Register.TORQUE]
#
#                 shoulder_vel = shoulder_vel if np.abs(shoulder_vel) > shoulder_thresh_vel else 0.0
#                 elbow_vel = elbow_vel if np.abs(elbow_vel) > elbow_thresh_vel else 0.0
#
#                 #shoulder_calc_meas_vel[index+1] = (shoulder_pos - shoulder_meas_pos[index]) / dt
#                 #elbow_calc_meas_vel[index+1] = (elbow_pos - elbow_meas_pos[index]) / dt
#
#
#                 # friction compensation
#                 if friction_compensation:
#                     # friction_regressor_mat = yb_friction_matrix(
#                     #    [shoulder_vel,
#                     #     elbow_vel])
#                     friction_regressor_mat = yb_friction_matrix(
#                         [shoulder_filtered_vel,
#                          elbow_filtered_vel])
#                     tau_fric = np.dot(friction_regressor_mat,
#                                       np.array(friction_terms))
#                     if np.abs(shoulder_filtered_vel) > 0.25:
#                         tau_fric[0] = np.clip(tau_fric[0], -1.0, 1.0)
#                     else:
#                         tau_fric[0] = 0.0
#                     if np.abs(elbow_filtered_vel) > 0.25:
#                         tau_fric[1] = np.clip(tau_fric[1], -1.0, 1.0)
#                     else:
#                         tau_fric[1] = 0.0
#
#                 # store the measured sensor data of
#                 # position, velocity and torque in each time step
#                 shoulder_meas_pos[index+1] = shoulder_pos
#                 shoulder_meas_vel[index+1] = shoulder_vel
#                 shoulder_meas_tau[index] = tau_cmd[0]
#                 elbow_meas_pos[index+1] = elbow_pos
#                 elbow_meas_vel[index+1] = elbow_vel
#                 elbow_meas_tau[index] = tau_cmd[1]
#                 shoulder_filtered_meas_vel[index+1] = shoulder_filtered_vel
#                 elbow_filtered_meas_vel[index+1] = elbow_filtered_vel
#
#                 # wait to enforce the demanded control frequency
#                 meas_dt = time.time() - start_loop
#                 if meas_dt > dt:
#                     print("Control loop is too slow!")
#                     print("Control frequency:", 1/meas_dt, "Hz")
#                     print("Desired frequency:", 1/dt, "Hz")
#                     print()
#                 while time.time() - start_loop < dt:
#                     pass
#
#                 # store times
#                 index += 1
#                 t += time.time() - start_loop
#                 meas_time[index+1] = t
#
#         except BaseException as e:
#             print("Exeception Block", e)
#
#         finally:
#             print("Disabling Motors...")
#             os.system(f"sudo -E env PATH=$PATH moteus_tool --stop -t{shoulder_id},{elbow_id}")
#             await sh_motor.set_stop()
#             await el_motor.set_stop()
#
#             date = datetime.now().strftime("%Y%m%d-%I%M%S-%p")
#             save_dir_time = os.path.join(save_dir, date)
#             if not os.path.exists(save_dir_time):
#                 os.makedirs(save_dir_time)
#             save_data(save_dir_time,
#                       date,
#                       shoulder_meas_pos[:index-1],
#                       shoulder_meas_vel[:index-1],
#                       shoulder_meas_tau[:index-1],
#                       elbow_meas_pos[:index-1],
#                       elbow_meas_vel[:index-1],
#                       elbow_meas_tau[:index-1],
#                       meas_time[:index-1])
#             plot_figure_single(save_dir=save_dir_time,
#                                date=date,
#                                index=index-1,
#                                meas_time=meas_time,
#                                shoulder_meas_pos=shoulder_meas_pos,
#                                shoulder_meas_vel=shoulder_meas_vel,
#                                shoulder_meas_tau=shoulder_meas_tau,
#                                elbow_meas_pos=elbow_meas_pos,
#                                elbow_meas_vel=elbow_meas_vel,
#                                elbow_meas_tau=elbow_meas_tau,
#                                shoulder_tau_controller=shoulder_tau_controller,
#                                elbow_tau_controller=elbow_tau_controller,
#                                shoulder_filtered_vel=shoulder_filtered_meas_vel,
#                                elbow_filtered_vel=elbow_filtered_meas_vel,
#                                shoulder_des_time=T_des,
#                                shoulder_des_pos=shoulder_des_pos,
#                                shoulder_des_vel=shoulder_des_vel,
#                                shoulder_des_tau=shoulder_des_tau,
#                                elbow_des_time=T_des,
#                                elbow_des_pos=elbow_des_pos,
#                                elbow_des_vel=elbow_des_vel,
#                                elbow_des_tau=elbow_des_tau,
#                                shoulder_fric_tau=shoulder_fric_tau,
#                                elbow_fric_tau=elbow_fric_tau,
#                                error=None)
#             date = plot_figure(save_dir=save_dir_time,
#                                date=date,
#                                index=index-1,
#                                meas_time=meas_time,
#                                shoulder_meas_pos=shoulder_meas_pos,
#                                shoulder_meas_vel=shoulder_meas_vel,
#                                shoulder_meas_tau=shoulder_meas_tau,
#                                elbow_meas_pos=elbow_meas_pos,
#                                elbow_meas_vel=elbow_meas_vel,
#                                elbow_meas_tau=elbow_meas_tau,
#                                shoulder_tau_controller=shoulder_tau_controller,
#                                elbow_tau_controller=elbow_tau_controller,
#                                shoulder_filtered_vel=shoulder_filtered_meas_vel,
#                                elbow_filtered_vel=elbow_filtered_meas_vel,
#                                shoulder_des_time=T_des,
#                                shoulder_des_pos=shoulder_des_pos,
#                                shoulder_des_vel=shoulder_des_vel,
#                                shoulder_des_tau=shoulder_des_tau,
#                                elbow_des_time=T_des,
#                                elbow_des_pos=elbow_des_pos,
#                                elbow_des_vel=elbow_des_vel,
#                                elbow_des_tau=elbow_des_tau,
#                                shoulder_fric_tau=shoulder_fric_tau,
#                                elbow_fric_tau=elbow_fric_tau,
#                                error=None)
#             # plt.figure()
#             # plt.plot(meas_time, shoulder_meas_vel)
#             # plt.plot(meas_time, shoulder_filtered_meas_vel)
#             # plt.plot(meas_time, shoulder_calc_meas_vel)
#             # plt.plot(meas_time, elbow_meas_vel)
#             # plt.plot(meas_time, elbow_filtered_meas_vel)
#             # plt.plot(meas_time, elbow_calc_meas_vel)
#             # plt.legend(["shoulder meas vel",
#             #             "shoulder vel filtered",
#             #             "shoulder calc vel",
#             #             "elbow meas vel",
#             #             "elbow vel filtered",
#             #             "elbow calc vel"])
#             # plt.show()
