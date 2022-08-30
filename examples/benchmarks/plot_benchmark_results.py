import os

from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

robot = "acrobot"

#base_dir = os.path.join("data", robot, "ilqr", "mpc_benchmark")
#base_dir = os.path.join("data", robot, "ilqr", "Kstab_benchmark")
#base_dir = os.path.join("data", robot, "ilqr", "mpc_benchmark_free")
base_dir = os.path.join("data", robot, "tvlqr_drake", "benchmark")
#base_dir = os.path.join("data", robot, "tvlqr", "benchmark")
#base_dir = os.path.join("data", robot, "pfl", "benchmark_collocated")

latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)

plot_benchmark_results(results_dir, costlim=[0, 5], show=True)
