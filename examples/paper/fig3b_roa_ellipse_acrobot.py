import pickle
import os
import multiprocessing
import numpy as np
import time
from matplotlib import patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


from double_pendulum.controller.lqr.roa.coopt_interface import caprr_coopt_interface
from double_pendulum.controller.lqr.roa.ellipsoid import getEllipseParamsFromQuad

# compute and plot roa for specific configurations
torque_limit = 5.0 # cmaes_controller_opt() implicitly assumes this
model1 = {"l1":0.3, "l2":0.2, "m1": 0.6, "m2": 0.6, "color":"blue"}
model2 = {"l1":0.2, "l2":0.3, "m1": 0.6, "m2": 0.6, "color":"red"}

# identity
opt_ctl_pars_acro  = {"Q":np.eye(4), "R":np.eye(2)}
opt_ctl_pars_pendu = {"Q":np.eye(4), "R":np.eye(2)}

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


# make ellipse plots
def make_comparison_plot(ax,configurations,robot):

    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11
    })

    x0 = np.pi
    x1 = 0


    for conf in configurations:
        if conf["robot"] == robot:
            w, h, a = getEllipseParamsFromQuad(0, 1, conf["rho"], conf["S"])
            l1 = conf["model"]["l1"]
            l2 = conf["model"]["l2"]
            lab = f"$l_1={l1}$, $l_2={l2}$"
            p = patches.Ellipse((x0, x1), w, h, a, alpha=1, ec=conf["model"]["color"], facecolor="none", label=lab)
            ax.add_patch(p)

    l = np.max([p.width, p.height])
    ax.set_xlim(x0-l/2, x0+l/2)
    ax.set_ylim(x1-l/2, x1+l/2)
    ax.grid(True)
    ax.set_xlabel("$q_1$")
    ax.set_ylabel("$q_2$")
    ax.legend()
    #ax.set_title(robot)
    #plt.show()

fig, ax = plt.subplots()
make_comparison_plot(ax,configurations,"acrobot")
plt.savefig("../../results/fig3b_acrobot_roa_ellipses")
#plt.show()
