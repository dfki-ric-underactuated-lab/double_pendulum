import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from double_pendulum.analysis.benchmark_scores import get_scores

design = "design_C.0"
model = "model_h1.1"

# data_dir = "data"
data_dir = "../../data/benchmarks"

scores = []
titles = []

# tvlqr
base_dir = os.path.join(data_dir, design, model, "acrobot", "tvlqr_drake", "benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")

base_dir = os.path.join(data_dir, design, model, "pendubot", "tvlqr_drake", "benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")

# ilqr mpc
base_dir = os.path.join(data_dir, design, model, "acrobot", "ilqr", "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR MPC (stab)")
print(ilqr_mpc_scores)

base_dir = os.path.join(data_dir, design, model, "pendubot", "ilqr", "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR MPC (stab)")

# ilqr K stab
base_dir = os.path.join(data_dir, design, model, "acrobot", "ilqr", "Kstab_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")

base_dir = os.path.join(data_dir, design, model, "pendubot", "ilqr", "Kstab_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")

# ilqr mpc free
base_dir = os.path.join(data_dir, design, model, "acrobot", "ilqr", "mpc_benchmark_free")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR MPC (free)")

base_dir = os.path.join(data_dir, design, model, "pendubot", "ilqr", "mpc_benchmark_free")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR MPC (free)")

# pfl col
base_dir = os.path.join(data_dir, design, model, "acrobot", "pfl", "benchmark_collocated")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL (col.)")

base_dir = os.path.join(data_dir, design, model, "pendubot", "pfl", "benchmark_collocated")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL (col.)")


SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 32

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rc('lines', linewidth=2.0)           # linewidth

n_rows = 2
n_cols = 5

fig, ax = plt.subplots(n_rows, n_cols, sharey="all", figsize=(16,6), squeeze=False)

for i, s in enumerate(scores):

    j = int(i % n_rows)
    k = int(i / n_rows)

    crits = ["model", r"$\dot{q}$ noise", r"$\tau$ noise", r"$\tau$ response", "delay"]
    numbers = [s["model"], s["measurement_noise"], s["u_noise"], s["u_responsiveness"], s["delay"]]
    bars = ax[j][k].bar(crits, numbers)
    bars[0].set_color("red")
    bars[1].set_color("blue")
    bars[2].set_color("green")
    bars[3].set_color("purple")
    bars[4].set_color("orange")
    if j == 0:
        ax[j][k].title.set_text(titles[i])
    ax[j][k].axes.xaxis.set_ticks([])
    #ax[i].title.set_text(titles[i])
    #ax[i].axes.xaxis.set_ticks([])

ax[0][0].set_ylabel("Acrobot\nScore")
ax[1][0].set_ylabel("Pendubot\nScore")
ax[0][0].set_ylim(0,1)
ax[1][0].set_ylim(0,1)
#ax[0][-1].legend(handles=bars, labels=crits, loc='upper left',
#     bbox_to_anchor=(1., 0.5), fancybox=False, shadow=False, ncol=1)
#ax[0][0].legend(handles=bars, labels=crits, loc='upper left',
#     bbox_to_anchor=(0.6, -0.002), fancybox=False, shadow=False, ncol=len(crits))
ax[1][0].legend(handles=bars, labels=crits, loc='upper left',
     bbox_to_anchor=(0.6, -0.002), fancybox=False, shadow=False, ncol=len(crits))


#plt.savefig(os.path.join(data_dir, design, model, "benchmark_scores.pdf"), bbox_inches="tight")
plt.tight_layout()
plt.show()
