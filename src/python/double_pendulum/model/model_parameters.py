import pprint
import yaml


class model_parameters():
    """
    Double pendulum plant parameters

    Parameters
    ----------
    mass : array_like, optional
        shape=(2,), dtype=float, default=[0.608, 0.630]
        masses of the double pendulum,
        [m1, m2], units=[kg]
    length : array_like, optional
        shape=(2,), dtype=float, default=[0.3, 0.4]
        link lengths of the double pendulum,
        [l1, l2], units=[m]
    com : array_like, optional
        shape=(2,), dtype=float, default=[0.275, 0.415]
        center of mass lengths of the double pendulum links
        [r1, r2], units=[m]
    damping : array_like, optional
        shape=(2,), dtype=float, default=[0.005, 0.005]
        damping coefficients of the double pendulum actuators
        [b1, b2], units=[kg*m/s]
    gravity : float, optional
        default=9.81
        gravity acceleration (pointing downwards),
        units=[m/s²]
    cfric : array_like, optional
        shape=(2,), dtype=float, default=[0.093, 0.14]
        coulomb friction coefficients for the double pendulum actuators
        [cf1, cf2], units=[Nm]
    inertia : array_like, optional
        shape=(2,), dtype=float, default=[0.0475, 0.0798]
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
        shape=(2,), dtype=float, default=[10.0, 10.0]
        torque limit of the motors
        [tl1, tl2], units=[Nm, Nm]
    dof : int, optional
        degrees of freedom of the double pendulum, default=2
        does not make sense to change
    filepath : string or path object, optional
        path to yaml file containing the the above parameters,
        if provided, the parameters from the yaml file will overwrite
        the other specified parameters
        default = None
    model_design : string, optional
        string description of the design to set the parameters for that
        specific design. model_design and model_id have to be set together.
        Options:
            - "A.0"
            - "B.0"
            - "C.0"
            - "hD.0"
        default=None
    model_id : string, optional
        string description of the model parameters for a design.
        Parameters for the specific model of the design will be
        loaded. model_design and model_id have to be set together.
        Options:
            - "1.0"
            - "1.1"
            - "2.0"
            - ...
        default=None
    robot : string, optional
        string describing the robot. Used to set the torque_limit when
        using model_design and model_id. Options:
            - "double_pendulum"
            - "acrobot"
            - "pendubot"
        default="double_pendulum"
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
                 filepath=None,
                 model_design=None,
                 model_id=None,
                 robot="double_pendulum"):

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

        if model_design is not None and model_id is not None:
            self.load_model(model_design, model_id, robot)

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

    def load_model(self, model_design, model_id, robot):
        if model_design == "design_A.0":
            if model_id[6] == "1":
                self.m = [0.608, 0.630]
                self.l = [0.3, 0.2]
                self.r = [0.275, 0.166]
                self.b = [0.081, 0.0]
                self.cf = [0.093, 0.186]
                self.g = 9.81
                self.I = [0.05472, 0.02522]
                self.Ir = 0.000060719
                self.gr = 6
                self.tl = [10., 10.]
                self.dof = 2
            elif model_id[6] == "2":
                self.m = [0.5476215952387185, 0.5978621372377623]
                self.l = [0.3, 0.2]
                self.r = [0.3, 0.2]
                self.b = [0.011078054769767294, 0.004396496974140761]
                self.cf = [0.09284176227841016, 0.07708291842936994]
                self.g = 9.81
                self.I = [0.053450262258304355, 0.024210710672023246]
                self.Ir = 6.287203962819607e-05
                self.gr = 6
                self.tl = [10., 10.]
                self.dof = 2

        if model_design == "design_B.0":
            if model_id[6] == "1":
                self.m = [0.6870144235621288, 0.6018697747980919]
                self.l = [0.3, 0.4]
                self.r = [0.3, 0.4150911636151641]
                self.b = [0.004999999999999999, 0.004999999999999999]
                self.cf = [0.09299999999999999, 0.13999999999999999]
                self.g = 9.81
                self.I = [0.047501752876886474, 0.07977851452810676]
                self.Ir = 8.823237128204706e-05
                self.gr = 6.0
                self.tl = [10.0, 10.0]
                self.dof = 2
            elif model_id[6:8] == "h2":
                self.m = [0.608, 0.63]
                self.l = [0.3, 0.4]
                self.r = [0.3, 0.4]
                self.b = [0.004999999999999999, 0.004999999999999999]
                self.cf = [0.093, 0.14]
                self.g = 9.81
                self.I = [0.05472, 0.10080000000000001]
                self.Ir = 8.823237128204706e-05
                self.gr = 6.0
                self.tl = [10.0, 10.0]
                self.dof = 2
        if model_design == "design_C.0":
            if model_id[6] == "3":
                self.m = [0.6416936775868083, 0.5639360564500752]
                self.l = [0.2, 0.3]
                self.r = [0.2, 0.32126693265850237]
                self.b = [0.001, 0.001]
                self.cf = [0.093, 0.078]
                self.g = 9.81
                self.I = [0.026710760905753486, 0.05387812962959988]
                self.Ir = 9.937281204851094e-05
                self.gr = 6.0
                self.tl = [10.0, 10.0]
                self.dof = 2
            elif model_id[6:8] == "h1":
                self.m = [0.5476215952387185, 0.5978621372377623]
                self.l = [0.2, 0.3]
                self.r = [0.2, 0.275]
                self.b = [0.011078054769767294, 0.004396496974140761]
                self.cf = [0.09284176227841016, 0.07708291842936994]
                self.g = 9.81
                self.I = [0.024210710672023246, 0.053450262258304355]
                self.Ir = 6.287203962819607e-05
                self.gr = 6.0
                self.tl = [10.0, 10.0]
                self.dof = 2
        if model_design == "design_hD.0":
            if model_id[6:8] == "h1":
                self.m = [0.5476215952387185, 0.5978621372377623]
                self.l = [0.3, 0.3]
                self.r = [0.3, 0.275]
                self.b = [0.011078054769767294, 0.004396496974140761]
                self.cf = [0.09284176227841016, 0.07708291842936994]
                self.g = 9.81
                self.I = [0.053450262258304355, 0.053450262258304355]
                self.Ir = 6.287203962819607e-05
                self.gr = 6.0
                self.tl = [10.0, 10.0]
                self.dof = 2

        if model_id[-2:] == ".1":
            self.b = [0., 0.]
            self.cf = [0., 0.]
            self.Ir = 0.
        elif model_id[-2:] == ".2":
            self.cf = [0., 0.]
            self.Ir = 0.

        if robot == "acrobot":
            self.tl[0] = 0.
        elif robot == "pendubot":
            self.tl[1] = 0.
