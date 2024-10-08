from copy import deepcopy

import numpy as np
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from scipy.integrate import odeint


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
    def __init__(
        self,
        plant,
        robustness,
        max_torque,
        robot,
        model="model_1.0",
    ):
        self.base_plant = deepcopy(plant)
        design = "design_C.1"

        model_par_path = (
            "../../../../data/system_identification/identified_parameters/"
            + design
            + "/"
            + model
            + "/model_parameters.yml"
        )
        mpar = model_parameters(filepath=model_par_path)
        torque_limit = [0.0, max_torque] if robot == "acrobot" else [max_torque, 0.0]
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

            for mp in mpar_vars:
                var = np.random.choice(modelpar_var_lists[mp])
                # this code could be further simplified by using setattr
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

    def odeint_integrator(self, y, dt, t, tau):
        # print('odeint t', t)
        plant_rhs = lambda y_, t_, u_: self.plant.rhs(t_, y_, u_)
        odeint_out = odeint(plant_rhs, y, [t, t + dt], args=(tau,))
        return odeint_out[1]

    def step(self, tau, dt, integrator="runge_kutta"):
        """
        Performs a simulation step with the specified integrator.
        Also adds process noise to the integration result.
        Uses and updates the internal state

        Parameters
        ----------
        tau : array_like, shape=(2,), dtype=float
            actuation input/motor torque,
            order=[u1, u2],
            units=[Nm]
        dt : float
            timestep, unit=[s]
        integrator : string
            string determining the integration method
            "euler" : Euler integrator
            "runge_kutta" : Runge Kutta integrator
             (Default value = "runge_kutta")
        """
        # tau = np.clip(
        #     tau,
        #     -np.asarray(self.plant.torque_limit),
        #     np.asarray(self.plant.torque_limit),
        # )

        if integrator == "runge_kutta":
            self.x = np.add(
                self.x,
                dt * self.runge_integrator(self.x, dt, self.t, tau),
                casting="unsafe",
            )
            # self.x += dt * self.runge_integrator(self.x, dt, self.t, tau)
        elif integrator == "euler":
            self.x = np.add(
                self.x,
                dt * self.euler_integrator(self.x, dt, self.t, tau),
                casting="unsafe",
            )
            # self.x += dt * self.euler_integrator(self.x, dt, self.t, tau)
        elif integrator == "odeint":
            self.x = self.odeint_integrator(self.x, dt, self.t, tau)
        else:
            raise NotImplementedError(
                f"Sorry, the integrator {integrator} is not implemented."
            )
        # process noise
        self.x = np.random.normal(self.x, self.process_noise_sigmas, np.shape(self.x))

        self.t += dt
        self.record_data(self.t, self.x.copy(), tau)
        # _ = self.get_measurement(dt)
