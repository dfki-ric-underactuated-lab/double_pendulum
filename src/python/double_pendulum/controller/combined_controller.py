

class CombinedController():
    def __init__(self,
                 controller1,
                 controller2,
                 condition1,
                 condition2):

        self.controllers = [controller1, controller2]
        self.active = 0

        self.conditions = [condition1, condition2]

    def init(self):
        self.controllers[0].init()
        self.controllers[1].init()

    def set_parameters(self, controller1_pars, controller2_pars):
        self.controllers[0].set_parameters(*controller1_pars)
        self.controllers[1].set_parameters(*controller2_pars)

    def set_start(self, x):
        self.controllers[0].set_start(x)
        self.controllers[1].set_start(x)

    def set_goal(self, x):
        self.controllers[0].set_goal(x)
        self.controllers[1].set_goal(x)

    def get_control_output(self, x, t):
        inactive = 1 - self.active

        if self.conditions[inactive](t, x):
            self.active = 1 - self.active
            print("Switching to Controller ", self.active + 1)

        return self.controllers[self.active].get_control_output(x, t)

    def get_forecast(self):
        return self.controllers[self.active].get_forecast()

    def get_init_trajectory(self):
        return self.controllers[self.active].get_init_trajectory()
