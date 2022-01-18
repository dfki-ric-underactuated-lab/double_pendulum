

class model_parameters():
    def __init__(self,
                 mass=[0.608, 0.630],
                 length=[0.3, 0.2],
                 com=[0.275, 0.166],
                 damping=[0.081, 0.0],
                 cfric=[0.093, 0.186],
                 gravity=9.81,
                 inertia=[0.05472, 0.2522],
                 torque_limit=[0.0, 10.0]):

        self.m = mass
        self.l = length
        self.com = com
        self.b = damping
        self.g = gravity
        self.coulomb_fric = cfric
        self.I = []
        self.torque_limit = torque_limit

        for i in range(self.dof):
            if inertia[i] is None:
                self.I.append(mass[i]*com[i]*com[i])
            else:
                self.I.append(inertia[i])
