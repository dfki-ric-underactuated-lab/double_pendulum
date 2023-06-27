from double_pendulum.analysis.leaderboard import get_swingup_time
from double_pendulum.utils.csv_trajectory import load_trajectory_full

data_dict = load_trajectory_full("data/ilqrfree/sim_swingup.csv")
T = data_dict["T"]
X = data_dict["X_meas"]
U = data_dict["U_con"]

time = get_swingup_time(T, X)
print(time)
