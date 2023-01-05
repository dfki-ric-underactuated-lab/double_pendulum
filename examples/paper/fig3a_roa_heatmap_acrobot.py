import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

robot = "acrobot"

base_dir = "../../results"
load_dir = os.path.join(base_dir, "design_optimization", robot, "lqr", "heatmaps")
load_filename = "heatmap_l1l2_acrobot.pickle"
load_file = os.path.join(load_dir, load_filename)

save_filename = "fig3a_acrobot_roa_heatmap.png"
save_file = os.path.join(base_dir, save_filename)

infile = open(load_file, 'rb')
results = pickle.load(infile, encoding='bytes')
prob_vols = -results["prob_vols"]
yticks = results["yticks"]
xticks = results["xticks"]

mark = [xticks[np.where(prob_vols == np.max(prob_vols))[1]], yticks[np.where(prob_vols == np.max(prob_vols))[0]]]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 26
})

fig = plt.figure(figsize=(8, 8))
# left right bottom top
plt.imshow(prob_vols, cmap='cividis', extent=[np.min(xticks), np.max(
    xticks), np.min(yticks), np.max(yticks)], origin="lower")
plt.colorbar(shrink=0.85, pad=0.005)
plt.scatter(0.2, 0.3, s=200, color="blue")
plt.scatter(0.3, 0.2, s=200, color="red")
#plt.text(mark[0]-0.01, mark[1]+0.01, "1", color="green")
plt.xlim(0.2, 0.4)
plt.xticks([0.2, 0.25, 0.3, 0.35, 0.4])
plt.xlabel("$l_2$ in $m$")
plt.ylabel("$l_1$ in $m$")
plt.savefig(save_file, bbox_inches="tight")
#plt.show()
