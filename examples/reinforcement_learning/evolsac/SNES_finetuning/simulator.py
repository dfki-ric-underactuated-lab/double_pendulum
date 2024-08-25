from copy import deepcopy

import numpy as np

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator


def get_par_list(x0, min_rel, max_rel, n):
    if x0 != 0:
        if n % 2 == 0:
            n = n + 1
        li = np.linspace(min_rel, max_rel, n)
    else:
        li = np.linspace(0, max_rel, n)
    par_list = li * x0
    return par_list


class CustomSimulator(Simulator):
    def __init__(self, plant, robustness, max_torque, robot):
        self.base_plant = deepcopy(plant)
        design = "design_C.1"
        model = "model_1.1"

        model_par_path = (
            "../../../../data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
        )
        mpar = model_parameters(filepath=model_par_path)
        torque_limit = [max_torque, 0] if robot == "pendubot" else [0, max_torque]
        mpar.set_torque_limit(torque_limit)

        self.mpar = mpar
        self.robustness = robustness
        self.max_torque = max_torque
        super().__init__(plant=plant)

    def reset(self):
        """
        Reset the Simulator
        Resets
            - the internal data recorder
            - the filter + arguments
            - the process noise
            - the measurement parameters
            - the motor parameters
            - perturbations
        """
        super().reset()

        self.plant = deepcopy(self.base_plant)

        if np.random.rand() < self.robustness:
            N_var = 21
            mpar_vars = [
                "Ir",
                "m1r1",
                "I1",
                "b1",
                "cf1",
                "m2r2",
                "m2",
                "I2",
                "b2",
                "cf2",
            ]

            Ir_var_list = np.linspace(0.0, 1e-4, N_var)
            m1r1_var_list = get_par_list(
                self.mpar.m[0] * self.mpar.r[0], 0.75, 1.25, N_var
            )
            I1_var_list = get_par_list(self.mpar.I[0], 0.75, 1.25, N_var)
            b1_var_list = np.linspace(-0.1, 0.1, N_var)
            cf1_var_list = np.linspace(-0.2, 0.2, N_var)
            m2r2_var_list = get_par_list(
                self.mpar.m[1] * self.mpar.r[1], 0.75, 1.25, N_var
            )
            m2_var_list = get_par_list(self.mpar.m[1], 0.75, 1.25, N_var)
            I2_var_list = get_par_list(self.mpar.I[1], 0.75, 1.25, N_var)
            b2_var_list = np.linspace(-0.1, 0.1, N_var)
            cf2_var_list = np.linspace(-0.2, 0.2, N_var)

            modelpar_var_lists = {
                "Ir": Ir_var_list,
                "m1r1": m1r1_var_list,
                "I1": I1_var_list,
                "b1": b1_var_list,
                "cf1": cf1_var_list,
                "m2r2": m2r2_var_list,
                "m2": m2_var_list,
                "I2": I2_var_list,
                "b2": b2_var_list,
                "cf2": cf2_var_list,
            }

            mp = np.random.choice(mpar_vars)
            var = np.random.choice(modelpar_var_lists[mp])

            if mp == "Ir":
                self.plant.Ir = var
            elif mp == "m1r1":
                m1 = self.mpar.m[0]
                r1 = var / m1
                self.plant.m[0] = m1
                self.plant.com[0] = r1
            elif mp == "I1":
                self.plant.I[0] = var
            elif mp == "b1":
                self.plant.b[0] = var
            elif mp == "cf1":
                self.plant.coulomb_fric[0] = var
            elif mp == "m2r2":
                m2 = self.mpar.m[1]
                r2 = var / m2
                self.plant.m[1] = m2
                self.plant.com[1] = r2
            elif mp == "m2":
                self.plant.m[1] = var
            elif mp == "I2":
                self.plant.I[1] = var
            elif mp == "b2":
                self.plant.b[1] = var
            elif mp == "cf2":
                self.plant.coulomb_fric[1] = var
