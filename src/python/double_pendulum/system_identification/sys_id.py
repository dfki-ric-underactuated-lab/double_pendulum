import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from double_pendulum.system_identification.dynamics import build_identification_matrices
from double_pendulum.system_identification.optimization import solve_least_squares
from double_pendulum.system_identification.plotting import plot_torques


def run_system_identification(measured_data_csv, g, n, L1, mp0, bounds):

    Q, phi = build_identification_matrices(g, n, L1, len(mp0), measured_data_csv)

    mp_opt = solve_least_squares(Q, phi, mp0, bounds)

    param_names = ["Lc1*m1", "I1", "Fc1", "Fv1", "Ir", "Lc2*m2", "m2", "I2", "Fc2", "Fv2"]
    print('Identified Parameters:')
    for i in range(len(param_names)):
        print("{:10s} = {:+.3e}".format(param_names[i], mp_opt[i]))

    # calculate errors
    Q_opt = phi.dot(mp_opt)
    mae = mean_absolute_error(Q.flatten(), Q_opt.flatten())
    rmse = mean_squared_error(Q.flatten(), Q_opt.flatten(), squared=False)

    print("Mean absolute error: ", mae)
    print("Mean root mean squared error: ", rmse)

    # plotting results
    data = pd.read_csv(measured_data_csv)
    time = data["time"].tolist()
    plot_torques(time, Q[::2, 0], Q[1::2, 0], Q_opt[::2], Q_opt[1::2])

    return mp_opt
