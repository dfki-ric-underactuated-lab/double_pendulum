import matplotlib.pyplot as plt

from double_pendulum.utils.pcw_polynomial import FitPiecewisePolynomial
from double_pendulum.utils.csv_trajectory import load_trajectory


init_csv_path = "../data/trajectories/acrobot/ilqr/trajectory.csv"
T, X, U = load_trajectory(init_csv_path)

t1 = 1.2
t2 = 3.4

y1_pcw = FitPiecewisePolynomial(data_x=T,
                                data_y=X.T[0],
                                num_break=40,
                                poly_degree=3)
y2_pcw = FitPiecewisePolynomial(data_x=T,
                                data_y=X.T[1],
                                num_break=40,
                                poly_degree=3)
fig, axs = plt.subplots(2, figsize=(15, 10))
for i in range(len(y1_pcw.x_sec_data)):
    axs[0].plot(y1_pcw.x_sec_data[i], y1_pcw.y_sec_data[i], '-', linewidth=3)
    axs[1].plot(y2_pcw.x_sec_data[i], y2_pcw.y_sec_data[i], '-', linewidth=3)
axs[0].plot(t1, y1_pcw.get_value(t1), '*', c='black',
            label=f'value at {t1}', markersize=15)
axs[1].plot(t2, y2_pcw.get_value(t2), '*', c='black',
            label=f'value at {t2}', markersize=15)
axs[0].set_ylabel('pos1')
axs[1].set_ylabel('pos2')
axs[1].set_xlabel('Time(Second)')
axs[0].set_title('Piecewise Polynomial')
for ax in axs:
    ax.legend()
plt.show()
