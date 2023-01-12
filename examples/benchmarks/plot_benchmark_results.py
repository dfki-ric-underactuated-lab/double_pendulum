import os

from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

design = "design_A.0"
model = "model_2.1"
robot = "pendubot"

# data_dir = "data"
data_dir = "../../data/benchmarks"

results_dir = os.path.join(data_dir, design, model, robot, "ilqr_stab")
#results_dir = os.path.join(data_dir, design, model, robot, "ilqr_riccati")
#results_dir = os.path.join(data_dir, design, model, robot, "ilqr_free")
#results_dir = os.path.join(data_dir, design, model, robot, "tvlqr_drake")
#results_dir = os.path.join(data_dir, design, model, robot, "tvlqr")
#results_dir = os.path.join(data_dir, design, model, robot, "pfl_collocated")

plot_benchmark_results(results_dir, costlim=[0, 5], show=True, save=False)
