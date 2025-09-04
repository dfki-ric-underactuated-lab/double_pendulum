# from https://stackoverflow.com/questions/46506850/how-can-i-get-input-from-an-xbox-one-controller-in-python
from evdev import ecodes, InputDevice, ff, util
import threading


# import math
# from inputs import get_gamepad
# import threading
#
#
# class GamePad(object):
#     """Gamepad Controller
#     Controller class to operate the double pendulum with a gamepad.
#
#     Parameters
#     ----------
#     gamepad_name: string
#         string refering to the gamepad type
#         Currently supported:
#             - "Logitech Logitech RumblePad 2 USB"
#         (Default value="Logitech Logitech RumblePad 2 USB")
#     """
#
#     def __init__(self, gamepad_name="Logitech Logitech RumblePad 2 USB"):
#
#         # self.MAX_TRIG_VAL = 1.  # math.pow(2, 8)
#         self.MAX_JOY_VAL = 255.0  # math.pow(2, 15)
#
#         self.LeftJoystickY = 0
#         self.LeftJoystickX = 0
#         self.RightJoystickY = 0
#         self.RightJoystickX = 0
#         self.LeftTrigger = 0
#         self.RightTrigger = 0
#         self.LeftBumper = 0
#         self.RightBumper = 0
#         self.A = 0
#         self.X = 0
#         self.Y = 0
#         self.B = 0
#         self.LeftThumb = 0
#         self.RightThumb = 0
#         self.Back = 0
#         self.Start = 0
#         self.LeftDPad = 0
#         self.RightDPad = 0
#         self.UpDPad = 0
#         self.DownDPad = 0
#
#         self._monitor_thread = threading.Thread(
#             target=self._monitor_controller, args=()
#         )
#         self._monitor_thread.daemon = True
#         self._monitor_thread.start()
#
#     def read(self):
#         """
#         Method to read the gamepad input.
#         """
#         x1 = self.LeftJoystickX
#         y1 = self.LeftJoystickY
#         x2 = self.RightJoystickX
#         y2 = self.RightJoystickY
#         # a = self.A
#         # b = self.X # b=1, x=2
#         # rb = self.RightBumper
#         return [x1, y1, x2, y2]
#
#     def _monitor_controller(self):
#         """
#         Method to monitor inputs from the gamepad.
#         """
#         while True:
#             events = get_gamepad()
#             for event in events:
#                 if event.code == "ABS_Y":
#                     self.LeftJoystickY = (
#                         event.state / self.MAX_JOY_VAL
#                     )  # normalize between -1 and 1
#                 if event.code == "ABS_X":
#                     self.LeftJoystickX = 2 * (
#                         event.state / self.MAX_JOY_VAL - 0.5
#                     )  # normalize between -1 and 1
#                 elif event.code == "ABS_Z":
#                     self.RightJoystickY = 2 * (
#                         event.state / self.MAX_JOY_VAL - 0.5
#                     )  # normalize between -1 and 1
#                 elif event.code == "ABS_RZ":
#                     self.RightJoystickX = 2 * (
#                         event.state / self.MAX_JOY_VAL - 0.5
#                     )  # normalize between -1 and 1
#                 # elif event.code == 'ABS_Z':
#                 #     self.LeftTrigger = event.state / GamePad.MAX_TRIG_VAL  # normalize between 0 and 1
#                 # elif event.code == 'ABS_RZ':
#                 #     self.RightTrigger = event.state / GamePad.MAX_TRIG_VAL  # normalize between 0 and 1
#                 # elif event.code == 'BTN_TL':
#                 #     self.LeftBumper = event.state
#                 # elif event.code == 'BTN_TR':
#                 #     self.RightBumper = event.state
#                 # elif event.code == 'BTN_SOUTH':
#                 #     self.A = event.state
#                 # elif event.code == 'BTN_NORTH':
#                 #     self.X = event.state
#                 # elif event.code == 'BTN_WEST':
#                 #     self.Y = event.state
#                 # elif event.code == 'BTN_EAST':
#                 #     self.B = event.state
#                 # elif event.code == 'BTN_THUMBL':
#                 #     self.LeftThumb = event.state
#                 # elif event.code == 'BTN_THUMBR':
#                 #     self.RightThumb = event.state
#                 # elif event.code == 'BTN_SELECT':
#                 #     self.Back = event.state
#                 # elif event.code == 'BTN_START':
#                 #     self.Start = event.state
#                 # elif event.code == 'BTN_TRIGGER_HAPPY1':
#                 #     self.LeftDPad = event.state
#                 # elif event.code == 'BTN_TRIGGER_HAPPY2':
#                 #     self.RightDPad = event.state
#                 # elif event.code == 'BTN_TRIGGER_HAPPY3':
#                 #     self.UpDPad = event.state
#                 # elif event.code == 'BTN_TRIGGER_HAPPY4':
#                 #     self.DownDPad = event.state
#
#
# if __name__ == "__main__":
#     joy = GamePad()
#     while True:
#         print(joy.read())
class GamePad(object):

    def __init__(self, gamepad_name="Logitech Logitech RumblePad 2 USB", dt=0.005):

        self.gamepad_name = gamepad_name

        # # get input device
        # for name in util.list_devices():
        #     self.dev = InputDevice(name)
        #     if self.dev.name == self.gamepad_name:
        #         break

        self.LeftJoystickX = 0.0
        self.LeftJoystickY = 0.0
        self.RightJoystickX = 0.0
        self.RightJoystickY = 0.0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.StartButton = 0
        self.BackButton = 0

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller, args=()
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        if gamepad_name == "Logitech Logitech RumblePad 2 USB":
            self.left_stick_x_name = ecodes.ABS_X
            self.left_bumper_name = ecodes.BTN_TOP2
            self.right_bumper_name = ecodes.BTN_PINKIE
            self.MAX_JOY_VAL = 255.0  # math.pow(2, 15)
        elif gamepad_name == "Logitech WingMan Cordless Gamepad":
            self.left_stick_x_name = ecodes.ABS_RZ
            self.left_bumper_name = ecodes.BTN_BASE
            self.right_bumper_name = ecodes.BTN_BASE2
        elif gamepad_name == "Logitech Gamepad F710":
            self.left_stick_x_name = ecodes.ABS_X
            self.left_stick_y_name = ecodes.ABS_Y
            self.right_stick_x_name = ecodes.ABS_RX
            self.right_stick_y_name = ecodes.ABS_RY
            self.left_bumper_name = ecodes.BTN_TL
            self.right_bumper_name = ecodes.BTN_TR
            self.start_button_name = ecodes.BTN_START
            self.back_button_name = ecodes.BTN_SELECT
            self.MAX_JOY_VAL = 2.0**15

        # get input device
        for name in util.list_devices():
            self.dev2 = InputDevice(name)
            if self.dev2.name == self.gamepad_name:
                break

        rumble_effect = ff.Rumble(strong_magnitude=65535, weak_magnitude=65535)
        effect_type = ff.EffectType(ff_rumble_effect=rumble_effect)
        duration_ms = int(dt * 1000)

        self.effect = ff.Effect(
            ecodes.FF_RUMBLE,  # type
            -1,  # id (set by ioctl)
            0,  # direction
            ff.Trigger(0, 0),  # no triggers
            ff.Replay(duration_ms, 0),  # length and delay
            ff.EffectType(ff_rumble_effect=rumble_effect),
        )

        self.rumble_on = False

    def read(self):
        # x = self.LeftJoystickX
        # lb = self.LeftBumper
        # rb = self.RightBumper
        # return x, lb, rb
        if self.gamepad_name == "Logitech Gamepad F710":
            xl = self.LeftJoystickX
            yl = -self.LeftJoystickY
            xr = self.RightJoystickX
            yr = -self.RightJoystickY

            start = self.StartButton
            back = self.BackButton
            l_bump = self.LeftBumper
            r_bump = self.RightBumper
        # return [
        #     self.LeftJoystickX,
        #     self.LeftJoystickY,
        #     self.RightJoystickX,
        #     self.RightJoystickY,
        # ]
        return [xl, yl, xr, yr, start, back, l_bump, r_bump]

    def _monitor_controller(self):
        while True:
            # get input device
            for name in util.list_devices():
                self.dev = InputDevice(name)
                if self.dev.name == self.gamepad_name:
                    break
            # read inputs
            for event in self.dev.read_loop():
                if event.type == ecodes.EV_ABS:
                    if event.code == self.left_stick_x_name:
                        # self.LeftJoystickX = 2 * (event.value / self.MAX_JOY_VAL - 0.5)
                        self.LeftJoystickX = event.value / self.MAX_JOY_VAL
                    if event.code == self.left_stick_y_name:
                        self.LeftJoystickY = event.value / self.MAX_JOY_VAL
                    if event.code == self.right_stick_x_name:
                        self.RightJoystickX = event.value / self.MAX_JOY_VAL
                    if event.code == self.right_stick_y_name:
                        self.RightJoystickY = event.value / self.MAX_JOY_VAL
                elif event.type == ecodes.EV_KEY:
                    if event.code == self.left_bumper_name:
                        self.LeftBumper = event.value
                    if event.code == self.right_bumper_name:
                        self.RightBumper = event.value
                    if event.code == self.start_button_name:
                        self.StartButton = event.value
                    if event.code == self.back_button_name:
                        self.BackButton = event.value

    def rumble(self):
        self.effect_id = self.dev2.upload_effect(self.effect)
        self.dev2.write(ecodes.EV_FF, self.effect_id, 1)
        self.rumble_on = True

    def stop_rumble(self):
        if self.rumble_on:
            self.dev2.erase_effect(self.effect_id)
            self.rumble_on = False


if __name__ == "__main__":
    joy = GamePad("Logitech Gamepad F710")
    while True:
        print(joy.read())
