import os
from datetime import datetime
import numpy as np

from double_pendulum.system_identification.sys_id import (run_system_identification,
                                                          run_system_identification_nl)
from double_pendulum.utils.plotting import plot_timeseries  # , plot_timeseries_csv
from double_pendulum.utils.csv_trajectory import save_trajectory, concatenate_trajectories


# saving
timestamp = datetime.today().strftime("%Y%m%d-%H%M%S")
save_dir = os.path.join("data", "system_identification", timestamp)
os.makedirs(save_dir)

# recorded data from robot
# measured_data_csv = [
#         "../data/system_identification/excitation_trajectories_measured/trajectory-pos-20_measured.csv",
#         #"../data/system_identification/excitation_trajectories_measured/trajectory-pos-50-3x_measured.csv"
#                     ]

# measured_data_csv = [
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-055032-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-055903-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-060245-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-060440-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-061705-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-055640-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-060143-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-060329-PM_measured.csv",
#     "../data/system_identification/recorded_data_20220812/sys_id/20220812-061157-PM_measured.csv"
#     ]
measured_data_csv = [
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_00.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_01.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_02.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_03.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_04.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_05.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_06.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_07.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_08.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_09.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_10.csv",  # exitation traj
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_11.csv",  # swingup traj
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_12.csv",
    "../../data/experiment_records/design_A.0/20220815/sys_id/trajectory_13.csv",
    ]


T, X, U = concatenate_trajectories(measured_data_csv,
                                   with_tau=True)
full_csv_path = os.path.join(save_dir, "full_trajectory.csv")
save_trajectory(full_csv_path, T, X, U)
plot_timeseries(T, X, U)

# fixed model parameters (will not be fitted)
fixed_mpar = {"g": 9.81,
              "gr": 6,
              "l1": 0.3,
              "l2": 0.2,
              "b1": 0.001,
              "b2": 0.001,
              "cf1": 0.19*2/np.pi,
              "cf2": 0.12*2/np.pi
             }

#variable_mpar = ["m1r1", "I1", "cf1", "b1", "Ir",
#                 "m2r2", "m2", "I2", "cf2", "b2"]
variable_mpar = ["m1r1", "I1", "Ir",
                 "m2r2", "m2", "I2"]

# initial model parameters
m1 = 0.608
m2 = 0.654
I1 = 0.05472
I2 = 0.10464
L2 = 0.2
Lc1 = 0.3
Lc2 = 0.2
Fc1 = 0.19*2/np.pi
Fv1 = 0.001
Fc2 = 0.12*2/np.pi
Fv2 = 0.001
Ir = 0.000060719

# m1 = 0.56
# m2 = 0.46
# I1 = 0.037
# I2 = 0.017
# L2 = 0.2
# Lc1 = 0.3
# Lc2 = 0.2
# Fc1 = 0.11
# Fv1 = 0.0
# Fc2 = 0.0
# Fv2 = 0.0085
# Ir = 1e-4

#mp0 = [Lc1 * m1, I1, Fc1, Fv1, Ir, Lc2 * m2, m2, I2, Fc2, Fv2]
mp0 = [Lc1 * m1, I1, Ir, Lc2 * m2, m2, I2]

# bounds = [[0.15, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 0.0, 0.00, 0.000],
#           [0.3, 1.0, 0.093, 0.005, 0.003, 0.4, 0.7, 1.0, 0.14, 0.005]]

bounds = np.array([[0.01, 0.5],      # r1*m1
                   [0.01, 0.2],       # I1
                   #[0.0, 0.5],       # cf1
                   #[0.0, 0.5],       # b1
                   [0.0, 0.003],     # Ir
                   [0.01, 0.5],       # r2*m2
                   [0.01, 1.0],       # m2
                   [0.01, 0.2],       # I2
                   #[0.0, 0.5],       # cf2
                   #[0.0, 0.5]        # b2
                  ]).T

# mpar_opt, mpar = run_system_identification(
#         measured_data_csv=measured_data_csv,
#         fixed_mpar=fixed_mpar,
#         variable_mpar=variable_mpar,
#         mp0=mp0,
#         bounds=bounds,
#         optimization_method="least-squares",
#         save_dir=save_dir,
#         num_proc=32,
#         rescale=False,
#         maxfevals=100000,
#         filt="butterworth")

mpar_opt, mpar = run_system_identification(
        measured_data_csv=measured_data_csv,
        fixed_mpar=fixed_mpar,
        variable_mpar=variable_mpar,
        mp0=mp0,
        bounds=bounds,
        optimization_method="least-squares",
        save_dir=save_dir,
        num_proc=32,
        rescale=True,
        maxfevals=100000,
        filt="butterworth")

print(mpar)

mpar.save_dict(os.path.join(save_dir, "model_parameters.yml"))
