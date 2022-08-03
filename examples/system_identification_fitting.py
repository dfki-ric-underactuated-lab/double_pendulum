import os
from datetime import datetime
import numpy as np

from double_pendulum.system_identification.sys_id import run_system_identification
from double_pendulum.utils.plotting import plot_timeseries  # , plot_timeseries_csv
from double_pendulum.utils.csv_trajectory import save_trajectory, concatenate_trajectories


# saving
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", "system_identification", timestamp)
os.makedirs(save_dir)

# recorded data from robot
measured_data_csv = [
        "../data/system_identification/excitation_trajectories_measured/trajectory-pos-20_measured.csv",
        "../data/system_identification/excitation_trajectories_measured/trajectory-pos-50-3x_measured.csv"
                    ]
read_withs = ["pandas", "pandas"]
keys = ["shoulder-elbow", "shoulder-elbow"]


full_csv_path = os.path.join(save_dir, "full_trajectory.csv")
T, X, U = concatenate_trajectories(measured_data_csv,
                                   read_withs=read_withs,
                                   with_tau=True,
                                   keys=keys,
                                   save_to=full_csv_path)
save_trajectory(full_csv_path, T, X, U)

# plot trajectory
# plot_timeseries_csv(full_csv_path, read_with="numpy", keys="")
plot_timeseries(T, X, U)



# fixed model parameters (will not be fitted)
fixed_mpar = {"g": 9.81,
              "gr": 6,
              "l1": 0.3,
              "l2": 0.4}

variable_mpar = ["m1r1", "I1", "cf1", "b1", "Ir",
                 "m2r2", "m2", "I2", "cf2", "b2"]

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

mp0 = [Lc1 * m1, I1, Fc1, Fv1, Ir, Lc2 * m2, m2, I2, Fc2, Fv2]

# bounds = [[0.15, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.0, 0.00, 0.000],
#           [0.3, 1.0, 0.093, 0.005, 0.003, 0.4, 0.7, 1.0, 0.14, 0.005]]

bounds = np.array([[0.15, 0.3],      # r1*m1
                   [0.0, 0.2],       # I1
                   [0.0, 0.5],       # cf1
                   [0.0, 0.5],       # b1
                   [0.0, 0.003],     # Ir
                   [0.1, 0.4],       # r2*m2
                   [0.5, 0.7],       # m2
                   [0.0, 0.2],       # I2
                   [0.0, 0.5],       # cf2
                   [0.0, 0.5]]).T    # b2

mpar_opt, mpar = run_system_identification(
        measured_data_csv=full_csv_path,
        fixed_mpar=fixed_mpar,
        variable_mpar=variable_mpar,
        mp0=mp0,
        bounds=bounds,
        read_with="numpy",
        keys="",
        optimization_method="least-squares",
        save_dir=save_dir,
        num_proc=0,
        rescale=False,
        maxfevals=100000)

print(mpar)

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
