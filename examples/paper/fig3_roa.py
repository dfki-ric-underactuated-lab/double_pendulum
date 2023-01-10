import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches

from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.controller.lqr.roa.ellipsoid import getEllipseParamsFromQuad

SMALL_SIZE = 20
MEDIUM_SIZE = 26
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

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

#ax[0][0].text(0.15, 0.22, "Acrobot", rotation=90)
ax[0][0].set_title("Acrobot RoA volume")
ax[0][1].set_title("Acrobot RoA ellipse")
ax[1][0].set_title("Pendubot RoA volume")
ax[1][1].set_title("Pendubot RoA ellipse")

D1_pars = np.loadtxt("../../results/design_optimization/pendubot/lqr/roa_designopt/model_par.csv")
D2_pars = np.loadtxt("../../results/design_optimization/acrobot/lqr/roa_designopt/model_par.csv")

##########
# top left
##########

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

divider = make_axes_locatable(ax[0][0])
cax = divider.append_axes('right', size='5%', pad=0.05)

# fig = plt.figure(figsize=(8, 8))
# # left right bottom top
im = ax[0][0].imshow(prob_vols, cmap='viridis', extent=[np.min(xticks), np.max(
    xticks), np.min(yticks), np.max(yticks)], origin="lower")


#fig.colorbar(im, cax=cax, shrink=0.85, pad=0.005, ticks=[0.25, 0.5, 0.75, 1.0, 1.25])
#fig.colorbar(im, cax=cax, shrink=0.85, pad=0.005, ticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
fig.colorbar(im, cax=cax, shrink=0.85, pad=0.005)
#plt.text(mark[0]-0.01, mark[1]+0.01, "1", color="green")
ax[0][0].plot(np.linspace(0.2, 0.3, 2), [0.2, 0.2], color="white")
ax[0][0].plot(np.linspace(0.2, 0.3, 2), [0.3, 0.3], color="white")
ax[0][0].plot([0.2, 0.2], np.linspace(0.2, 0.3, 2), color="white")
ax[0][0].plot([0.3, 0.3], np.linspace(0.2, 0.3, 2), color="white")
ax[0][0].scatter(D1_pars[2], D1_pars[1], s=200, color="blue", zorder=5)
ax[0][0].scatter(D2_pars[2], D2_pars[1], s=200, color="red", zorder=5)
ax[0][0].set_xlim(0.1, 0.4)
ax[0][0].set_ylim(0.1, 0.4)
ax[0][0].set_xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
ax[0][0].set_yticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
ax[0][0].set_xlabel("$l_2$ [$m$]")
ax[0][0].set_ylabel("$l_1$ [$m$]")
#ax[0][0].savefig(save_file, bbox_inches="tight")

###########
# top right
###########

def get_semi_minor(S, rho):
    s = S / rho
    ev = np.linalg.eigvals(s)
    axes = ev**-(1/2)
    minor = np.min(axes)
    return minor

# compute and plot roa for specific configurations
torque_limit = 5.0 # cmaes_controller_opt() implicitly assumes this
#model1 = {"l1":0.3, "l2":0.2, "m1": 0.6, "m2": 0.6, "color":"blue"}
#model2 = {"l1":0.2, "l2":0.3, "m1": 0.6, "m2": 0.6, "color":"red"}
model1 = {"l1":D1_pars[1], "l2":D1_pars[2], "m1": 0.6, "m2": 0.6, "color":"blue"}
model2 = {"l1":D2_pars[1], "l2":D2_pars[2], "m1": 0.6, "m2": 0.6, "color":"red"}

opt_ctl_pars_acro  = {"Q":np.eye(4), "R":np.eye(2)}
opt_ctl_pars_pendu = {"Q":np.eye(4), "R":np.eye(2)}
#opt_ctl_pars_acro  = {"Q":np.diag((D1_pars[0], D1_pars[1], D1_pars[2], D1_pars[3])), "R":np.diag((D1_pars[4], D1_pars[4]))}
#opt_ctl_pars_pendu  = {"Q":np.diag((D2_pars[0], D2_pars[1], D2_pars[2], D2_pars[3])), "R":np.diag((D2_pars[4], D2_pars[4]))}

configurations = [{"robot":"acrobot",  "model":model1, "opt_ctl_pars":opt_ctl_pars_acro, "rho":None, "S":None, "vol":None},
                  {"robot":"acrobot",  "model":model2, "opt_ctl_pars":opt_ctl_pars_acro, "rho":None, "S":None, "vol":None}]

backend = "sos_con" # or "najafi"
nevals = 100000 #for najafi

for idx,conf in enumerate(configurations):

    # create plant
    if conf["robot"] == "pendubot":
        tau_max = [torque_limit, 0.0]
    elif conf["robot"] == "acrobot":
        tau_max =[0.0, torque_limit]

    model = conf["model"]
    design_params = {
        "m": [model["m1"],  model["m2"]],
        "l": [model["l1"],  model["l2"]],
        "lc": [model["l1"], model["l2"]],
        "b": [0.0, 0.0],
        "fc": [0.0, 0.0],
        "g": 9.81,
        "I": [model["m1"]*model["l1"]**2, model["m2"]*model["l2"]**2],
        "tau_max": tau_max}

    Q = conf["opt_ctl_pars"]["Q"]
    R = conf["opt_ctl_pars"]["R"]

    # do a Roa final ROA estimation and store data for plotting
    caprr_roaEst = caprr_coopt_interface(
        design_params, Q, R, backend=backend,
        najafi_evals=nevals, estimate_clbk=None, robot=conf["robot"])

    vol, rhof, S = caprr_roaEst._estimate()
    conf["vol"]=vol
    conf["rho"]=rhof
    conf["S"]=S
    conf["minor_axis"] = get_semi_minor(S, rhof)
    print(conf)

# make ellipse plots
def make_comparison_plot(ax,configurations,robot):

    # plt.rcParams.update({
    # "text.usetex": True,
    # "font.family": "serif",
    # "font.size": 26
    # })

    x0 = np.pi
    x1 = 0


    for conf in configurations:
        if conf["robot"] == robot:
            w, h, a = getEllipseParamsFromQuad(0, 1, conf["rho"], conf["S"])
            l1 = conf["model"]["l1"]
            l2 = conf["model"]["l2"]
            lab = f"$l_1={np.around(l1, 2)}$, $l_2={np.around(l2, 2)}$"
            p = patches.Ellipse((x0, x1), w, h, a, alpha=1, ec=conf["model"]["color"], facecolor="none", label=lab)
            ax.add_patch(p)

    l = np.max([p.width, p.height])
    if robot == "acrobot":
        ax.set_xlim(x0-l/3, x0+l/3)
    else:
        ax.set_xlim(x0-l/2, x0+l/2)
    ax.set_ylim(x1-l/2, x1+l/2)
    ax.grid(True)
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.legend()
    #ax.set_title(robot)
    #plt.show()

make_comparison_plot(ax[0][1], configurations, "acrobot")

#############
# bottom left
#############

robot = "pendubot"
base_dir = "../../results"
load_dir = os.path.join(base_dir, "design_optimization", robot, "lqr", "heatmaps")
load_filename = "heatmap_l1l2_pendubot.pickle"
load_file = os.path.join(load_dir, load_filename)

save_filename = "fig3c_pendubot_roa_heatmap.png"
save_file = os.path.join(base_dir, save_filename)

infile = open(load_file, 'rb')
results = pickle.load(infile, encoding='bytes')
prob_vols = -results["prob_vols"]
yticks = results["yticks"]
xticks = results["xticks"]

mark = [xticks[np.where(prob_vols == np.max(prob_vols))[1]], yticks[np.where(prob_vols == np.max(prob_vols))[0]]]

divider = make_axes_locatable(ax[1][0])
cax = divider.append_axes('right', size='5%', pad=0.05)

# fig = plt.figure(figsize=(8, 8))
# # left right bottom top
im = ax[1][0].imshow(prob_vols, cmap='viridis', extent=[np.min(xticks), np.max(
    xticks), np.min(yticks), np.max(yticks)], origin="lower")


#fig.colorbar(im, cax=cax, shrink=0.85, pad=0.005, ticks=[0.01, 0.02, 0.03, 0.04])
fig.colorbar(im, cax=cax, shrink=0.85, pad=0.005, ticks=[0.02, 0.04, 0.06, 0.08])
#fig.colorbar(im, cax=cax, shrink=0.85, pad=0.005)
#plt.text(mark[0]-0.01, mark[1]+0.01, "1", color="green")
ax[1][0].plot(np.linspace(0.2, 0.3, 2), [0.2, 0.2], color="white")
ax[1][0].plot(np.linspace(0.2, 0.3, 2), [0.3, 0.3], color="white")
ax[1][0].plot([0.2, 0.2], np.linspace(0.2, 0.3, 2), color="white")
ax[1][0].plot([0.3, 0.3], np.linspace(0.2, 0.3, 2), color="white")
ax[1][0].scatter(D1_pars[2], D1_pars[1], s=200, color="blue", zorder=5)
ax[1][0].scatter(D2_pars[2], D2_pars[1], s=200, color="red", zorder=5)
ax[1][0].set_xlim(0.1, 0.4)
ax[1][0].set_ylim(0.1, 0.4)
ax[1][0].set_xticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
ax[1][0].set_yticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
ax[1][0].set_xlabel("$l_2$ [$m$]")
ax[1][0].set_ylabel("$l_1$ [$m$]")

##############
# bottom right
##############

torque_limit = 5.0 # cmaes_controller_opt() implicitly assumes this
model1 = {"l1":D1_pars[1], "l2":D1_pars[2], "m1": 0.6, "m2": 0.6, "color":"blue"}
model2 = {"l1":D2_pars[1], "l2":D2_pars[2], "m1": 0.6, "m2": 0.6, "color":"red"}

opt_ctl_pars_acro  = {"Q":np.eye(4), "R":np.eye(2)}
opt_ctl_pars_pendu = {"Q":np.eye(4), "R":np.eye(2)}
#opt_ctl_pars_acro  = {"Q":np.diag((D1_pars[0], D1_pars[1], D1_pars[2], D1_pars[3])), "R":np.diag((D1_pars[4], D1_pars[4]))}
#opt_ctl_pars_pendu  = {"Q":np.diag((D2_pars[0], D2_pars[1], D2_pars[2], D2_pars[3])), "R":np.diag((D2_pars[4], D2_pars[4]))}

configurations = [
                  {"robot":"pendubot", "model":model1, "opt_ctl_pars":opt_ctl_pars_pendu, "rho":None, "S":None, "vol":None},
                  {"robot":"pendubot", "model":model2, "opt_ctl_pars":opt_ctl_pars_pendu, "rho":None, "S":None, "vol":None}]

backend = "sos_con" # or "najafi"
nevals = 100000 #for najafi

for idx,conf in enumerate(configurations):

    # create plant
    if conf["robot"] == "pendubot":
        tau_max = [torque_limit, 0.0]
    elif conf["robot"] == "acrobot":
        tau_max =[0.0, torque_limit]

    model = conf["model"]
    design_params = {
        "m": [model["m1"],  model["m2"]],
        "l": [model["l1"],  model["l2"]],
        "lc": [model["l1"], model["l2"]],
        "b": [0.0, 0.0],
        "fc": [0.0, 0.0],
        "g": 9.81,
        "I": [model["m1"]*model["l1"]**2, model["m2"]*model["l2"]**2],
        "tau_max": tau_max}

    Q = conf["opt_ctl_pars"]["Q"]
    R = conf["opt_ctl_pars"]["R"]

    # do a Roa final ROA estimation and store data for plotting
    caprr_roaEst = caprr_coopt_interface(
        design_params, Q, R, backend=backend,
        najafi_evals=nevals, estimate_clbk=None, robot=conf["robot"])

    vol, rhof, S = caprr_roaEst._estimate()
    conf["vol"]=vol
    conf["rho"]=rhof
    conf["S"]=S
    conf["minor_axis"] = get_semi_minor(S, rhof)
    print(conf)


make_comparison_plot(ax[1][1], configurations, "pendubot")

###########

plt.subplots_adjust(left=0.08,
                    bottom=0.08,
                    right=0.98,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.25)
plt.savefig("../../results/fig3_roa", bbox_inches="tight")
#plt.show()
