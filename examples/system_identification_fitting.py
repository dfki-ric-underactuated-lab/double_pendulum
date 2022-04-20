import numpy as np

from double_pendulum.system_identification.sys_id import run_system_identification
from double_pendulum.utils.plotting import plot_timeseries_csv


# recorded data from robot
measured_data_csv = "../data/system_identification/excitation_trajectories_measured/trajectory-pos-50-3x_measured.csv"
#plot_timeseries_csv(measured_data_csv, read_with="pandas")

# fixed model parameters (will not be fitted)
fixed_mpar = {"g": 9.81,
              "gr": 6,
              "l1": 0.3}

variable_mpar = ["m1r1", "I1", "cf1", "b1", "Ir", "m2r2", "m2", "I2", "cf2", "b2"]

# initial model parameters
m1 = 0.608
m2 = 0.654
I1 = 0.05472
I2 = 0.10464
L2 = 0.4
Lc1 = 0.3
Lc2 = 0.4
Fc1 = 0.093
Fv1 = 0.005
Fc2 = 0.14
Fv2 = 0.0008
Ir = 0.000060719
# Irr = 0.002186

mp0 = [Lc1 * m1,  I1, Fc1, Fv1, Ir, Lc2 * m2, m2, I2, Fc2, Fv2]

bounds = ([0.15, 0.0, 0.08, 0.004, 0.0, 0.1, 0.5, 0.0, 0.12, 0.0005],
          [0.3, np.Inf, 0.093, 0.005, 0.003, 0.4, 0.7, np.Inf, 0.14, 0.0008])

mpar_opt = run_system_identification(measured_data_csv=measured_data_csv,
                                     fixed_mpar=fixed_mpar,
                                     variable_mpar=variable_mpar,
                                     mp0=mp0,
                                     bounds=bounds)
