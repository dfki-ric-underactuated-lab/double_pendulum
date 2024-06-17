import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from double_pendulum.analysis.benchmark_scores import get_scores

design = "design_C.0"
model = "model_3.1"
robot = "acrobot"

scores = []
titles = []

# data_dir = "data"
data_dir = "../../data/benchmarks"

# tvlqr
results_dir = os.path.join(data_dir, design, model, robot, "tvlqr_drake")
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")

# ilqr mpc
results_dir = os.path.join(data_dir, design, model, robot, "ilqr_stab")
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR MPC (stab)")

# ilqr K stab
results_dir = os.path.join(data_dir, design, model, robot, "ilqr_riccati")
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR MPC (Riccati)")

# ilqr mpc free
results_dir = os.path.join(data_dir, design, model, robot, "ilqr_free")
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR MPC (free)")

# pfl col
results_dir = os.path.join(data_dir, design, model, robot, "pfl_collocated")
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL (col.)")

# scores = [
#           tvlqr_scores,
#           ilqr_mpc_scores,
#           ilqr_Kstab_scores,
#           ilqr_free_scores,
#           pfl_col_scores]
# titles = ["TVLQR", "iLQR traj", "iLQR Riccati", "iLQR free", "PFL col."]


SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 32

mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
mpl.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
mpl.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
mpl.rc("lines", linewidth=2.0)  # linewidth

n_rows = 1
n_cols = 5
# fig, ax = plt.subplots(n_rows, n_cols, sharey="all", figsize=(16,9), squeeze=False)
fig, ax = plt.subplots(n_rows, n_cols, sharey="all", figsize=(16, 4), squeeze=False)

for i, s in enumerate(scores):
    j = int(i % n_rows)
    k = int(i / n_rows)

    crits = []
    if "model" in s.keys():
        crits.append("model")
    if "measurement_noise" in s.keys():
        crits.append(r"$\dot{q}$ noise")
    if "u_noise" in s.keys():
        crits.append(r"$\tau$ noise")
    if "u_responsiveness" in s.keys():
        crits.append(r"$\tau$ response")
    if "delay" in s.keys():
        crits.append("delay")
    if "perturbation" in s.keys():
        crits.append("pert.")
    numbers = []
    for key in s.keys():
        numbers.append(s[key])
    bars = ax[j][k].bar(crits, numbers)
    colors = ["red", "blue", "green", "purple", "orange", "magenta"]
    for i in range(len(crits)):
        bars[i].set_color(colors[i])
    ax[j][k].title.set_text(titles[i])
    ax[j][k].axes.xaxis.set_ticks([])
    # ax[i].title.set_text(titles[i])
    # ax[i].axes.xaxis.set_ticks([])

ax[0][0].set_ylabel("Score")
# ax[1][0].set_ylabel("Score")
ax[0][0].set_ylim(0, 1)
# ax[1][0].set_ylim(0,1)
# ax[0][-1].legend(handles=bars, labels=crits, loc='upper left',
#     bbox_to_anchor=(1., 0.5), fancybox=False, shadow=False, ncol=1)
ax[0][0].legend(
    handles=bars,
    labels=crits,
    loc="upper left",
    bbox_to_anchor=(0.6, -0.002),
    fancybox=False,
    shadow=False,
    ncol=len(crits),
)

# plt.savefig(os.path.join(data_dir, robot, robot+"_benchmark_scores.pdf"), bbox_inches="tight")
plt.tight_layout()
plt.show()
