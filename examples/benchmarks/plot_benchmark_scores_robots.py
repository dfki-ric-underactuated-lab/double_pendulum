import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from double_pendulum.analysis.benchmark_scores import get_scores

design = "design_C.0"
model = "model_3.1"

data_dir = "../../data/benchmarks"

scores = []
titles = []

# tvlqr
results_dir = os.path.join(data_dir, design, model, "acrobot", "tvlqr_drake")
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")

results_dir = os.path.join(data_dir, design, model, "pendubot", "tvlqr_drake")
tvlqr_scores = get_scores(results_dir)
scores.append(tvlqr_scores)
titles.append("TVLQR")

# ilqr mpc
results_dir = os.path.join(data_dir, design, model, "acrobot", "ilqr_stab")
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR MPC (stab)")

results_dir = os.path.join(data_dir, design, model, "pendubot", "ilqr_stab")
ilqr_mpc_scores = get_scores(results_dir)
scores.append(ilqr_mpc_scores)
titles.append("iLQR MPC (stab)")

# ilqr K stab
results_dir = os.path.join(data_dir, design, model, "acrobot", "ilqr_riccati")
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")

results_dir = os.path.join(data_dir, design, model, "pendubot", "ilqr_riccati")
ilqr_Kstab_scores = get_scores(results_dir)
scores.append(ilqr_Kstab_scores)
titles.append("iLQR (Riccati)")

# ilqr mpc free
results_dir = os.path.join(data_dir, design, model, "acrobot", "ilqr_free")
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR MPC (free)")

results_dir = os.path.join(data_dir, design, model, "pendubot", "ilqr_free")
ilqr_free_scores = get_scores(results_dir)
scores.append(ilqr_free_scores)
titles.append("iLQR MPC (free)")

# pfl col
results_dir = os.path.join(data_dir, design, model, "acrobot", "pfl_collocated")
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL (col.)")

results_dir = os.path.join(data_dir, design, model, "pendubot", "pfl_collocated")
pfl_col_scores = get_scores(results_dir)
scores.append(pfl_col_scores)
titles.append("PFL (col.)")


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

n_rows = 2
n_cols = 5

fig, ax = plt.subplots(n_rows, n_cols, sharey="all", figsize=(16, 6), squeeze=False)

for i, s in enumerate(scores):
    j = int(i % n_rows)
    k = int(i / n_rows)

    # crits = ["model", r"$\dot{q}$ noise", r"$\tau$ noise", r"$\tau$ response", "delay"]
    # numbers = [s["model"], s["measurement_noise"], s["u_noise"], s["u_responsiveness"], s["delay"]]
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
    bars[0].set_color("red")
    bars[1].set_color("blue")
    bars[2].set_color("green")
    bars[3].set_color("purple")
    bars[4].set_color("orange")
    if j == 0:
        ax[j][k].title.set_text(titles[i])
    ax[j][k].axes.xaxis.set_ticks([])
    # ax[i].title.set_text(titles[i])
    # ax[i].axes.xaxis.set_ticks([])

ax[0][0].set_ylabel("Acrobot\nScore")
ax[1][0].set_ylabel("Pendubot\nScore")
ax[0][0].set_ylim(0, 1)
ax[1][0].set_ylim(0, 1)
# ax[0][-1].legend(handles=bars, labels=crits, loc='upper left',
#     bbox_to_anchor=(1., 0.5), fancybox=False, shadow=False, ncol=1)
# ax[0][0].legend(handles=bars, labels=crits, loc='upper left',
#     bbox_to_anchor=(0.6, -0.002), fancybox=False, shadow=False, ncol=len(crits))
ax[1][0].legend(
    handles=bars,
    labels=crits,
    loc="upper left",
    bbox_to_anchor=(0.6, -0.002),
    fancybox=False,
    shadow=False,
    ncol=len(crits),
)


# plt.savefig(os.path.join(data_dir, design, model, "benchmark_scores.pdf"), bbox_inches="tight")
plt.tight_layout()
plt.show()
