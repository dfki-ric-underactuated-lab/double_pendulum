import pprint
import yaml


class model_parameters():
    """
    Double pendulum plant parameters

    Parameters
    ----------
    mass : array_like, optional
        shape=(2,), dtype=float, default=[1.0, 1.0]
        masses of the double pendulum,
        [m1, m2], units=[kg]
    length : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.5]
        link lengths of the double pendulum,
        [l1, l2], units=[m]
    com : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.5]
        center of mass lengths of the double pendulum links
        [r1, r2], units=[m]
    damping : array_like, optional
        shape=(2,), dtype=float, default=[0.5, 0.5]
        damping coefficients of the double pendulum actuators
        [b1, b2], units=[kg*m/s]
    gravity : float, optional
        default=9.81
        gravity acceleration (pointing downwards),
        units=[m/s²]
    coulomb_fric : array_like, optional
        shape=(2,), dtype=float, default=[0.0, 0.0]
        coulomb friction coefficients for the double pendulum actuators
        [cf1, cf2], units=[Nm]
    inertia : array_like, optional
        shape=(2,), dtype=float, default=[None, None]
        inertia of the double pendulum links
        [I1, I2], units=[kg*m²]
        if entry is None defaults to point mass m*l² inertia for the entry
    motor_inertia : float, optional
        default=0.0
        inertia of the actuators/motors
        [Ir1, Ir2], units=[kg*m²]
    gear_ratio : int, optional
        gear ratio of the motors, default=6
    torque_limit : array_like, optional
        shape=(2,), dtype=float, default=[np.inf, np.inf]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    dof : int, optional
        degrees of freedom of the double pendulum, default=2
        does not make sense to change
    filepath : string or path object
        path to yaml file containing the the above parameters,
        if provided, the parameters from the yaml file will overwrite
        the other specified parameters
    """
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
        """
        setter function for mass

        Parameters
        ----------
        mass : array_like, optional
            shape=(2,), dtype=float, default=[1.0, 1.0]
            masses of the double pendulum,
            [m1, m2], units=[kg]
        """
        self.m = mass

    def set_length(self, length):
        """
        setter function for length

        Parameters
        ----------
        length : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            link lengths of the double pendulum,
            [l1, l2], units=[m]
        """
        self.l = length

    def set_com(self, com):
        """
        setter function for com

        Parameters
        ----------
        com : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            center of mass lengths of the double pendulum links
            [r1, r2], units=[m]
        """
        self.r = com

    def set_damping(self, damping):
        """
        setter function for damping

        Parameters
        ----------
        damping : array_like, optional
            shape=(2,), dtype=float, default=[0.5, 0.5]
            damping coefficients of the double pendulum actuators
            [b1, b2], units=[kg*m/s]
        """
        self.b = damping

    def set_cfric(self, cfric):
        """
        setter function for coulomb friction

        Parameters
        ----------
        cfric : array_like, optional
            shape=(2,), dtype=float, default=[0.0, 0.0]
            coulomb friction coefficients for the double pendulum actuators
            [cf1, cf2], units=[Nm]
        """
        self.cf = cfric

    def set_gravity(self, gravity):
        """
        setter function for gravity

        Parameters
        ----------
        gravity : float, optional
            default=9.81
            gravity acceleration (pointing downwards),
            units=[m/s²]
        """
        self.g = gravity

    def set_inertia(self, inertia):
        """
        setter function for inertia

        Parameters
        ----------
        inertia : array_like, optional
            shape=(2,), dtype=float, default=[None, None]
            inertia of the double pendulum links
            [I1, I2], units=[kg*m²]
            if entry is None defaults to point mass m*l² inertia for the entry
        """
        self.I = inertia

    def set_motor_inertia(self, motor_inertia):
        """
        setter function for motor inertia

        Parameters
        ----------
        motor_inertia : float, optional
            default=0.0
            inertia of the actuators/motors
            [Ir1, Ir2], units=[kg*m²]
        """
        self.Ir = motor_inertia

    def set_gear_ratio(self, gear_ratio):
        """
        setter function for gear ratio

        Parameters
        ----------
        gear_ratio : int, optional
            gear ratio of the motors, default=6
        """
        self.gr = gear_ratio

    def set_torque_limit(self, torque_limit):
        """
        setter function for torque limit

        Parameters
        ----------
        torque_limit : array_like, optional
            shape=(2,), dtype=float, default=[np.inf, np.inf]
            torque limit of the motors
            [tl1, tl2], units=[Nm, Nm]
        """
        self.tl = torque_limit

    def set_dof(self, dof):
        """
        setter function for degrees of freedom

        Parameters
        ----------
        dof : int, optional
            degrees of freedom of the double pendulum, default=2
            does not make sense to change
        """
        self.dof = dof

    def get_dict(self):
        """
        get dictionary with all parameters
        """
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
        """
        save all parameters in a yaml file
        keys used are:
            - m1, m2
            - l1, l2
            - r1, r2
            - b1, b2
            - cf1, cf2
            - g
            - I1, I2
            - Ir
            - gr
            - tl1, tl2

        Parameters
        ----------
        save_path : string or path object
            path where the yaml file will be stored
        """
        mpar_dict = self.get_dict()
        with open(save_path, 'w') as f:
            yaml.dump(mpar_dict, f)

    def load_dict(self, mpar_dict):
        """
        load parameters from a dictionary

        Parameters
        ----------
        mpar_dict : dict
            dictionary containing the parameters
            keys which are looked for are:
                - m1, m2
                - l1, l2
                - r1, r2
                - b1, b2
                - cf1, cf2
                - g
                - I1, I2
                - Ir
                - gr
                - tl1, tl2
        """
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
        """
        load parameters from a yaml file

        Parameters
        ----------
        file_path : string or path object
            path to yaml file containing the the above parameters,
            if provided, the parameters from the yaml file will overwrite
            the other specified parameters
        """
        with open(file_path, 'r') as f:
            mpar_dict = yaml.safe_load(f)
        self.load_dict(mpar_dict)

    def __str__(self):
        mpar_dict = self.get_dict()
        return pprint.pformat(mpar_dict)

    def __repr__(self):
        return self.__str__()
