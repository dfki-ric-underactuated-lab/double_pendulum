import yaml
import pprint


class model_parameters():
    def __init__(self,
                 mass=[0.608, 0.630],
                 length=[0.3, 0.4],
                 com=[0.275, 0.415],
                 damping=[0.005, 0.005],
                 cfric=[0.093, 0.14],
                 gravity=9.81,
                 inertia=[0.0475, 0.0798],
                 motor_inertia=0.,
                 gear_ratio=6,
                 torque_limit=[10.0, 10.0],
                 dof=2,
                 filepath=None):

        self.m = mass
        self.l = length
        self.r = com
        self.b = damping
        self.cf = cfric
        self.g = gravity
        self.I = []
        self.Ir = motor_inertia
        self.gr = gear_ratio
        self.tl = torque_limit
        self.dof = dof

        for i in range(self.dof):
            if inertia[i] is None:
                self.I.append(mass[i]*com[i]*com[i])
            else:
                self.I.append(inertia[i])

        if filepath is not None:
            self.load_yaml(filepath)

    def set_mass(self, mass):
        self.m = mass

    def set_length(self, length):
        self.l = length

    def set_com(self, com):
        self.r = com

    def set_damping(self, damping):
        self.b = damping

    def set_cfric(self, cfric):
        self.cf = cfric

    def set_gravity(self, gravity):
        self.g = gravity

    def set_inertia(self, inertia):
        self.I = inertia

    def set_motor_inertia(self, motor_inertia):
        self.Ir = motor_inertia

    def set_gear_ratio(self, gear_ratio):
        self.gr = gear_ratio

    def set_torque_limit(self, torque_limit):
        self.tl = torque_limit

    def set_dof(self, dof):
        self.dof = dof

    def get_dict(self):
        mpar_dict = {}
        mpar_dict["m1"] = float(self.m[0])
        mpar_dict["m2"] = float(self.m[1])
        mpar_dict["l1"] = float(self.l[0])
        mpar_dict["l2"] = float(self.l[1])
        mpar_dict["r1"] = float(self.r[0])
        mpar_dict["r2"] = float(self.r[1])
        mpar_dict["b1"] = float(self.b[0])
        mpar_dict["b2"] = float(self.b[1])
        mpar_dict["cf1"] = float(self.cf[0])
        mpar_dict["cf2"] = float(self.cf[1])
        mpar_dict["g"] = float(self.g)
        mpar_dict["I1"] = float(self.I[0])
        mpar_dict["I2"] = float(self.I[1])
        mpar_dict["Ir"] = float(self.Ir)
        mpar_dict["gr"] = float(self.gr)
        mpar_dict["tl1"] = float(self.tl[0])
        mpar_dict["tl2"] = float(self.tl[1])
        return mpar_dict

    def save_dict(self, save_path):
        mpar_dict = self.get_dict()
        with open(save_path, 'w') as f:
            yaml.dump(mpar_dict, f)

    def load_dict(self, mpar_dict):
        if "m1" in mpar_dict.keys() and "m2" in mpar_dict.keys():
            self.m = [mpar_dict["m1"], mpar_dict["m2"]]
        if "l1" in mpar_dict.keys() and "l2" in mpar_dict.keys():
            self.l = [mpar_dict["l1"], mpar_dict["l2"]]
        if "r1" in mpar_dict.keys() and "r2" in mpar_dict.keys():
            self.r = [mpar_dict["r1"], mpar_dict["r2"]]
        if "b1" in mpar_dict.keys() and "b2" in mpar_dict.keys():
            self.b = [mpar_dict["b1"], mpar_dict["b2"]]
        if "cf1" in mpar_dict.keys() and "cf2" in mpar_dict.keys():
            self.cf = [mpar_dict["cf1"], mpar_dict["cf2"]]
        if "g" in mpar_dict.keys():
            self.g = mpar_dict["g"]
        if "I1" in mpar_dict.keys() and "I2" in mpar_dict.keys():
            self.I = [mpar_dict["I1"], mpar_dict["I2"]]
        if "Ir" in mpar_dict.keys():
            self.Ir = mpar_dict["Ir"]
        if "gr" in mpar_dict.keys():
            self.gr = mpar_dict["gr"]
        if "tl1" in mpar_dict.keys() and "tl2" in mpar_dict.keys():
            self.tl = [mpar_dict["tl1"], mpar_dict["tl2"]]

    def load_yaml(self, file_path):
        with open(file_path, 'r') as f:
            mpar_dict = yaml.safe_load(f)
        self.load_dict(mpar_dict)

    def __str__(self):
        mpar_dict = self.get_dict()
        return pprint.pformat(mpar_dict)

    def __repr__(self):
        return self.__str__()
