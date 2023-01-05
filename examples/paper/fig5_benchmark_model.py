#!/usr/bin/python3

import os
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from double_pendulum.model.model_parameters import model_parameters

robot = "acrobot"
#version = "v1.0"
design = "design_A.0"
model = "model_2.1"

#base_dir = os.path.join("data", robot, version, "ilqr", "mpc_benchmark")
results_dir = os.path.join("../../results/benchmarks", design, model, robot, "ilqr_stab")
# base_dir = os.path.join("data", robot, version, "ilqr", "Kstab_benchmark")
# base_dir = os.path.join("data", robot, version, "ilqr", "mpc_benchmark_free")
# base_dir = os.path.join("data", robot, version, "tvlqr_drake", "benchmark")
# base_dir = os.path.join("data", robot, version, "pfl", "benchmark_collocated")

costlim = [0., 2.]

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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
    #"font.size": 26
})

# load data
pickle_path = os.path.join(results_dir, "results.pkl")
f = open(pickle_path, 'rb')
res_dict = pickle.load(f)
f.close()

fig_counter = 0

# ToDo: solve better
if "meas_noise_robustness" in res_dict.keys():
    if "free_costs" in res_dict["meas_noise_robustness"]["None"].keys():
        y1 = np.median(res_dict["meas_noise_robustness"]["None"]["free_costs"], axis=1)
        norm_cost_free = y1[0]
    # if "following_costs" in res_dict["meas_noise_robustness"]["None"].keys():
    #     y2 = np.median(res_dict["meas_noise_robustness"]["None"]["following_costs"], axis=1)
    #     norm_cost_follow = y2[0]
else:
    norm_cost_free = 1.
    norm_cost_follow = 1.

order = [0, 1,
         4, 6,
         8, 3,
         2, 5,
         7, 9]

# model robustness
if "model_robustness" in res_dict.keys():

    mpar = model_parameters()
    mpar.load_yaml(os.path.join(results_dir, "model_parameters.yml"))
    mpar_dict = mpar.get_dict()

    fig_mr, ax_mr = plt.subplots(2, 5, figsize=(20, 4),
                                 num=fig_counter)
    #fig_mr.suptitle("Model Robustness")
    for i, mp in enumerate(res_dict["model_robustness"].keys()):

        ii = order[i]

        j = int(ii % 2)
        k = int(ii / 2)
        # ax[j][k].set_title(f"{mp}")

        ymax = 0.

        x = res_dict["model_robustness"][mp]["values"]
        if "free_costs" in res_dict["model_robustness"][mp].keys():
            y1 = np.asarray(res_dict["model_robustness"][mp]["free_costs"]) / norm_cost_free
            ax_mr[j][k].plot(x, y1, "o")
            ymax = np.max([ymax, np.max(y1)])
        # if "following_costs" in res_dict["model_robustness"][mp].keys():
        #     y2 = np.asarray(res_dict["model_robustness"][mp]["following_costs"]) / norm_cost_follow
        #     ax_mr[j][k].plot(x, y2, "o-")
        #     ymax = np.max([ymax, np.max(y2)])

        if costlim is not None:
            ymax = min(costlim[1], 1.1*ymax)
        # ymax = np.max([np.max(y1), np.max(y2)])  # costlim[1]

        if "successes" in res_dict["model_robustness"][mp].keys():
            xr = x[:-1] + 0.5*np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            succ = res_dict["model_robustness"][mp]["successes"]
            for ii in range(len(xr[:-1])):
                c = "red"
                if succ[ii]:
                    c = "green"
                ax_mr[j][k].add_patch(
                        Rectangle((xr[ii], 0.),
                                  xr[ii+1]-xr[ii], ymax,
                                  facecolor=c, edgecolor=None,
                                  alpha=0.1))
        if mp == "m1r1":
            temp = mpar_dict["m1"]*mpar_dict["r1"]
            mpar_x = [temp, temp]
        elif mp == "m2r2":
            temp = mpar_dict["m2"]*mpar_dict["r2"]
            mpar_x = [temp, temp]
        else:
            mpar_x = [mpar_dict[mp], mpar_dict[mp]]
        mpar_y = [0, ymax]
        ax_mr[j][k].plot(mpar_x, mpar_y, "--", color="grey")

        #if costlim is not None:
        #    ax_mr[j][k].set_ylim(costlim[0], costlim[1])
        if ymax > 2:
            ymin = 0.
        else:
            ymin = 0.95
        ax_mr[j][k].set_ylim(ymin, ymax)
        #if k == 0:
        if i == 0:
            #ax_mr[j][k].set_ylabel("rel. cost")
            ax_mr[j][k].text(-0.00004, 0.4, "relative cost", rotation="vertical", fontsize=20)
        if mp == "Ir":
            xlab = r"$I_r$"
        elif mp == "m2r2":
            xlab = r"$m_2 r_2$"
        elif mp == "m1r1":
            xlab = r"$m_1 r_1$"
        elif mp == "m2":
            xlab = r"$m_2$"
        elif mp == "I1":
            xlab = r"$I_1$"
        elif mp == "I2":
            xlab = r"$I_2$"
        elif mp == "b1":
            xlab = r"$b_1$"
        elif mp == "b2":
            xlab = r"$b_2$"
        elif mp == "cf1":
            xlab = r"$c_{f1}$"
        elif mp == "cf2":
            xlab = r"$c_{f2}$"

        ax_mr[j][k].set_xlabel(xlab)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.5)
    plt.savefig(os.path.join("../../results", "fig5_benchmark_model_robustness.pdf"),
                bbox_inches="tight")
    #plt.show()
