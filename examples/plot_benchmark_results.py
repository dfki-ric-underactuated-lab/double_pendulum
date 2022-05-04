import os

from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

robot = "acrobot"
method = "ilqr"

base_dir = os.path.join("data", robot, method, "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)

plot_benchmark_results(results_dir, show=True)
