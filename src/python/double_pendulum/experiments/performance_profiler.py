import cProfile
import pstats
import time
import numpy as np

from motor_driver.canmotorlib import CanMotorController

def motor_send_n_commands(numTimes=1000):
    global motor1, motor2
    for i in range(numTimes):
        pos1, vel1, curr1 = motor1.send_deg_command(0, 0, 0, 0, 0)
        pos2, vel2, curr2 = motor2.send_deg_command(0, 0, 0, 0, 0)

def motor_send_n_commands2(motor_ids=[8, 9], can_port="can0", numTimes=1000):
    motor1 = CanMotorController(can_port, motor_ids[0])
    motor2 = CanMotorController(can_port, motor_ids[1])
    motor1.enable_motor()
    motor2.enable_motor()
    for i in range(numTimes):
        pos1, vel1, curr1 = motor1.send_deg_command(0, 0, 0, 0, 0)
        pos2, vel2, curr2 = motor2.send_deg_command(0, 0, 0, 0, 0)
    motor1.disable_motor()
    motor2.disable_motor()

def motor_speed_test(motor_ids=[8, 9], can_port="can0", n=1000):
    global motor1, motor2

    motor1 = CanMotorController(can_port, motor_ids[0])
    motor2 = CanMotorController(can_port, motor_ids[1])
    motor1.enable_motor()
    motor2.enable_motor()
    startdtTest = time.time()
    print("Starting Profiler for {} Commands".format(n))
    cmdString = "motor_send_n_commands({})".format(int(n))
    profiler = cProfile.Profile()
    profiler.run(cmdString)
    profiler.print_stats()

    enddtTest = time.time()
    motor1.disable_motor()
    motor2.disable_motor()
    dt = (enddtTest - startdtTest) / n
    cmd_freq = 1 / dt
    print("Dt = {}".format(dt))
    print("Command Frequency: {} Hz".format(cmd_freq))


def motor_speed_test2(motor_ids=[8, 9], can_port="can0", n=1000):
    profiler = cProfile.Profile()
    profiler.enable()
    motor_send_n_commands2(motor_ids, can_port, n)
    profiler.disable()
    return pstats.Stats(profiler)

def profiler(data_dict, start, end, meas_dt):
    """
    validate avg dt of the control loop with (start time - end time) / numSteps
    """
    n = data_dict["n"]
    dt = data_dict["dt"]
    des_time = data_dict["des_time_list"]
    meas_time = data_dict["meas_time_list"]

    meas_avg_dt = (end - start)/n

    # Compare desired versus measured control frequency
    print("Performance Profiler:")
    print(
        f"{'Desired frequency:  ':>40}{(1/dt):<20}{' Hz':<20}",
        f"\n{'Measured avg frequency:  ':>40}{(1/meas_avg_dt):<20}{' Hz':<20}",
        # f"\n{'Measured sample frequency:  ':>40}{(1/meas_dt):<20}{'
        # Hz':<20}",
        f"\n{'Difference in Meas-Des frequency:  ':>40}"
        f"{(abs((1/dt)-(1/meas_avg_dt))):<20}{' Hz':<20}",
    )
    print()
    print(
        f"{'Desired dt:  ':>40}{dt:<21}{' s':<20}",
        f"\n{'Measured avg dt:  ':>40}{meas_avg_dt:<21}{' s':<20}",
        # f"\n{'Measured sample dt:  ':>40}{meas_dt:<21}{' s':<20}",
        f"\n{'Time error Meas-Des dt:  ':>40}{abs((dt-meas_avg_dt)*1000):<21}{' milliseconds':<20}",
    )
    print()
    print("Time total desired:", des_time[n - 1], "s")
    print("Time total measured:", meas_time[n - 1], "s")
    # print("meas_time:", len(meas_time))
    # print("des_time:", len(des_time))
    # print("meas_pos:", len(meas_pos))
    # print("des_pos:", len(des_pos))
    print()
