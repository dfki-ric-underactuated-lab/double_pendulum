import numpy as np


class CombinedController():
    def __init__(self,
                 controller1,
                 controller2,
                 condition1,
                 condition2,
                 compute_both=False):

        self.controllers = [controller1, controller2]
        self.active = 0

        self.conditions = [condition1, condition2]

        self.compute_both = compute_both

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

        if self.compute_both:
            _ = self.controllers[inactive].get_control_output(x, t)

        return self.controllers[self.active].get_control_output(x, t)

    def get_forecast(self):
        return self.controllers[self.active].get_forecast()

    def get_init_trajectory(self):
        return self.controllers[self.active].get_init_trajectory()


class SimultaneousControllers():
    def __init__(self,
                 controllers,
                 forecast_con=0):

        self.controllers = controllers
        self.fc_ind = forecast_con

    def init(self):
        for c in self.controllers:
            c.init()

    def set_parameters(self, controller_pars):
        for i, c in enumerate(self.controllers):
            c.set_parameters(*(controller_pars[i]))

    def set_start(self, x):
        for c in self.controllers:
            c.set_start(x)

    def set_goal(self, x):
        for c in self.controllers:
            c.set_goal(x)

    def get_control_output(self, x, t):
        u_cons = []
        for c in self.controllers:
            u_cons.append(c.get_control_output(x, t))

        u = np.sum(u_cons)
        return u

    def get_forecast(self):
        return self.controllers[self.fc_ind].get_forecast()

    def get_init_trajectory(self):
        return self.controllers[self.fc_ind].get_init_trajectory()

