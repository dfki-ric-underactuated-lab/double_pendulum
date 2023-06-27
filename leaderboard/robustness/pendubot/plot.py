import argparse

from double_pendulum.analysis.benchmark_plot import plot_benchmark_results

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-dir",
    dest="save_dir",
    help="Directory for loading and saving data.",
    default="data",
    required=True,
)

save_dir = parser.parse_args().save_dir

plot_benchmark_results(
    save_dir,
    "benchmark_results.pkl",
    costlim=[0, 5],
    show=False,
    save=True,
    file_format="png",
    scale=0.5,
)
