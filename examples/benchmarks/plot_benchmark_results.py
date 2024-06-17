import os

from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

design = "design_C.0"
model = "model_3.1"
robot = "acrobot"

# data_dir = "data"
data_dir = "../../data/benchmarks"
date_folder = False


results_dir = os.path.join(data_dir, design, model, robot, "ilqr_stab")
# results_dir = os.path.join(data_dir, design, model, robot, "ilqr_riccati")
# results_dir = os.path.join(data_dir, design, model, robot, "ilqr_free")
# results_dir = os.path.join(data_dir, design, model, robot, "tvlqr_drake")
# results_dir = os.path.join(data_dir, design, model, robot, "tvlqr")
# results_dir = os.path.join(data_dir, design, model, robot, "pfl_collocated")

if date_folder:
    results_dir = os.path.join(results_dir, sorted(os.listdir(results_dir))[-1])

print("loading data from:", results_dir)

plot_benchmark_results(results_dir, costlim=[0, 5], show=True, save=False)
