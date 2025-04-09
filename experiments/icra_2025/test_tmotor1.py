import sys
from double_pendulum.experiments.experimental_utils import setZeroPosition
from motor_driver.canmotorlib import CanMotorController
import time

if len(sys.argv) < 2:
    print("Please provide the motor id as an argument")
    exit()


can_port = "can0"
motor_id = int(sys.argv[1])
# motor_type = "AK80_6_V1p1"
# motor_type = "AK80_9_V1p1"
motor_type = "AK80_9_V2"

# create
motor_controller = CanMotorController(can_port, motor_id, motor_type)

# enable
pos, vel, torque = motor_controller.enable_motor()

# set zero
print("Setting zero position")
setZeroPosition(motor_controller, pos)
print("Done")

# Send tau command to motors
for i in range(5000):
    print("sending command ...")
    pos, vel, tau = motor_controller.send_rad_command(0.0, 0.0, 0.0, 0.0, 0.0)
    time.sleep(0.002)

# disable
pos, vel, tau = motor_controller.disable_motor()
