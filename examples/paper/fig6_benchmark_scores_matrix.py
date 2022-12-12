import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from double_pendulum.analysis.benchmark_scores import get_scores

design1 = "design_A.0"
model1 = "model_2.1"

design2 = "design_C.0"
model2 = "model_h1.1"

scores = []
titles = []
model_titles = []

# data_dir = "data"
data_dir = "../../data/benchmarks"

#model_str1 = "A.0 2.1"
#model_str2 = "C.0 1.1"
model_str1 = r"$\mathbb{M}_1$"
model_str2 = r"$\mathbb{M}_2$"

# tvlqr
base_dir = os.path.join(data_dir, design1, model1, "acrobot", "tvlqr_drake", "benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")
model_titles.append(model_str1)


base_dir = os.path.join(data_dir, design1, model1, "pendubot", "tvlqr_drake", "benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design2, model2, "acrobot", "tvlqr_drake", "benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")
model_titles.append(model_str2)

base_dir = os.path.join(data_dir, design2, model2, "pendubot", "tvlqr_drake", "benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")
model_titles.append(model_str2)

# ilqr mpc
base_dir = os.path.join(data_dir, design1, model1, "acrobot", "ilqr", "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR (stab)")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design1, model1, "pendubot", "ilqr", "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR (stab)")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design2, model2, "acrobot", "ilqr", "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR (stab)")
model_titles.append(model_str2)

base_dir = os.path.join(data_dir, design2, model2, "pendubot", "ilqr", "mpc_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR (stab)")
model_titles.append(model_str2)

# ilqr K stab
base_dir = os.path.join(data_dir, design1, model1, "acrobot", "ilqr", "Kstab_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design1, model1, "pendubot", "ilqr", "Kstab_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design2, model2, "acrobot", "ilqr", "Kstab_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")
model_titles.append(model_str2)

base_dir = os.path.join(data_dir, design2, model2, "pendubot", "ilqr", "Kstab_benchmark")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")
model_titles.append(model_str2)

# ilqr mpc free
base_dir = os.path.join(data_dir, design1, model1, "acrobot", "ilqr", "mpc_benchmark_free")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR (free)")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design1, model1, "pendubot", "ilqr", "mpc_benchmark_free")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR (free)")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design2, model2, "acrobot", "ilqr", "mpc_benchmark_free")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR (free)")
model_titles.append(model_str2)

base_dir = os.path.join(data_dir, design2, model2, "pendubot", "ilqr", "mpc_benchmark_free")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR (free)")
model_titles.append(model_str2)

# pfl col
base_dir = os.path.join(data_dir, design1, model1, "acrobot", "pfl", "benchmark_collocated")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design1, model1, "pendubot", "pfl", "benchmark_collocated")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL")
model_titles.append(model_str1)

base_dir = os.path.join(data_dir, design2, model2, "acrobot", "pfl", "benchmark_collocated")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL")
model_titles.append(model_str2)

base_dir = os.path.join(data_dir, design2, model2, "pendubot", "pfl", "benchmark_collocated")
latest_dir = sorted(os.listdir(base_dir))[-1]
results_dir = os.path.join(base_dir, latest_dir)
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL")
model_titles.append(model_str2)


plt.style.use("../../data/other/latexstyle.mplstyle")

SMALL_SIZE = 26
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rc('lines', linewidth=2.0)           # linewidth
mpl.rc('text', usetex=True)
mpl.rc('font', family="serif")

n_rows = 2
n_cols = 10

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
        ax[j][k].title.set_text(model_titles[i])
    ax[j][k].axes.xaxis.set_ticks([])

ax[0][0].set_ylabel("Acrobot\nScore")
ax[1][0].set_ylabel("Pendubot\nScore")
ax[0][0].set_ylim(0,1)
ax[1][0].set_ylim(0,1)
ax[1][0].legend(handles=bars, labels=crits, loc='upper left',
     bbox_to_anchor=(-1., -0.002), fancybox=False, shadow=False, ncol=len(crits))

ax[0][0].text(2.5, 1.2, titles[0], fontsize=MEDIUM_SIZE)
ax[0][2].text(0.0, 1.2, titles[4], fontsize=MEDIUM_SIZE)
ax[0][4].text(-0.2, 1.2, titles[8], fontsize=MEDIUM_SIZE)
ax[0][6].text(1., 1.2, titles[12], fontsize=MEDIUM_SIZE)
ax[0][8].text(3.5, 1.2, titles[16], fontsize=MEDIUM_SIZE)

plt.savefig(os.path.join("figures", "fig6_benchmark_scores_matrix.pdf"), bbox_inches="tight")
plt.tight_layout()
plt.show()
