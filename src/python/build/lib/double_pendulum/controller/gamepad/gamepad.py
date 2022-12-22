# from https://stackoverflow.com/questions/46506850/how-can-i-get-input-from-an-xbox-one-controller-in-python

import math
from inputs import get_gamepad
import threading


class GamePad(object):
    """Gamepad Controller
    Controller class to operate the double pendulum with a gamepad.

    Parameters
    ----------
    gamepad_name: string
        string refering to the gamepad type
        Currently supported:
            - "Logitech Logitech RumblePad 2 USB"
        (Default value="Logitech Logitech RumblePad 2 USB")
    """

    def __init__(self, gamepad_name="Logitech Logitech RumblePad 2 USB"):

        # self.MAX_TRIG_VAL = 1.  # math.pow(2, 8)
        self.MAX_JOY_VAL = 255.  # math.pow(2, 15)

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):
        """ 
        Method to read the gamepad input.
        """
        x1 = self.LeftJoystickX
        #y1 = self.LeftJoystickY
        #x2 = self.RightJoystickX
        y2 = self.RightJoystickY
        #a = self.A
        #b = self.X # b=1, x=2
        #rb = self.RightBumper
        return [x1, y2]

    def _monitor_controller(self):
        """ 
        Method to monitor inputs from the gamepad.
        """
        while True:
            events = get_gamepad()
            for event in events:
                #if event.code == 'ABS_Y':
                #    self.LeftJoystickY = event.state / self.MAX_JOY_VAL  # normalize between -1 and 1
                if event.code == 'ABS_X':
                    self.LeftJoystickX = 2*(event.state / self.MAX_JOY_VAL - 0.5)  # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.RightJoystickY = 2*(event.state / self.MAX_JOY_VAL - 0.5)  # normalize between -1 and 1
                #elif event.code == 'ABS_RZ':
                #    self.RightJoystickX = 2*(event.state / self.MAX_JOY_VAL - 0.5)  # normalize between -1 and 1
                # elif event.code == 'ABS_Z':
                #     self.LeftTrigger = event.state / GamePad.MAX_TRIG_VAL  # normalize between 0 and 1
                # elif event.code == 'ABS_RZ':
                #     self.RightTrigger = event.state / GamePad.MAX_TRIG_VAL  # normalize between 0 and 1
                # elif event.code == 'BTN_TL':
                #     self.LeftBumper = event.state
                # elif event.code == 'BTN_TR':
                #     self.RightBumper = event.state
                # elif event.code == 'BTN_SOUTH':
                #     self.A = event.state
                # elif event.code == 'BTN_NORTH':
                #     self.X = event.state
                # elif event.code == 'BTN_WEST':
                #     self.Y = event.state
                # elif event.code == 'BTN_EAST':
                #     self.B = event.state
                # elif event.code == 'BTN_THUMBL':
                #     self.LeftThumb = event.state
                # elif event.code == 'BTN_THUMBR':
                #     self.RightThumb = event.state
                # elif event.code == 'BTN_SELECT':
                #     self.Back = event.state
                # elif event.code == 'BTN_START':
                #     self.Start = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY1':
                #     self.LeftDPad = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY2':
                #     self.RightDPad = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY3':
                #     self.UpDPad = event.state
                # elif event.code == 'BTN_TRIGGER_HAPPY4':
                #     self.DownDPad = event.state


if __name__ == '__main__':
    joy = GamePad()
    while True:
        print(joy.read())
