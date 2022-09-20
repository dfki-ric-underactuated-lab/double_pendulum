import os

from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

design = "design_A.0"
model = "model_2.1"
robot = "pendubot"

# data_dir = "data"
data_dir = "../../data/benchmarks"

base_dir = os.path.join(data_dir, design, model, robot, "ilqr", "mpc_benchmark")
#base_dir = os.path.join(data_dir, design, model, robot, "ilqr", "Kstab_benchmark")
#base_dir = os.path.join(data_dir, design, model, robot, "ilqr", "mpc_benchmark_free")
#base_dir = os.path.join(data_dir, design, model, robot, "tvlqr_drake", "benchmark")
#base_dir = os.path.join(data_dir, design, model, robot, "tvlqr", "benchmark")
#base_dir = os.path.join(data_dir, design, model, robot, "pfl", "benchmark_collocated")

latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)

plot_benchmark_results(results_dir, costlim=[0, 5], show=True)
