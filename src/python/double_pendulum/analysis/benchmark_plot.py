import os
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.analysis.benchmark_scores import get_scores


def plot_benchmark_results(
    results_dir,
    filename="results.pkl",
    costlim=[0, 1e6],
    show=False,
    save=True,
    file_format="pdf",
    scale=1.0,
):
    SMALL_SIZE = 20 * scale
    MEDIUM_SIZE = 24 * scale
    BIGGER_SIZE = 32 * scale

    mpl.rc("font", size=SMALL_SIZE)  # controls default text sizes
    mpl.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    mpl.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    mpl.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    mpl.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    mpl.rc("lines", linewidth=2.0 * scale)  # linewidth

    # load data
    pickle_path = os.path.join(results_dir, filename)
    f = open(pickle_path, "rb")
    res_dict = pickle.load(f)
    f.close()

    fig_counter = 0

    # ToDo: solve better
    if "meas_noise_robustness" in res_dict.keys():
        if "free_costs" in res_dict["meas_noise_robustness"]["None"].keys():
            y1 = np.median(
                res_dict["meas_noise_robustness"]["None"]["free_costs"], axis=1
            )
            norm_cost_free = y1[0]
        # if "following_costs" in res_dict["meas_noise_robustness"]["None"].keys():
        #     y2 = np.median(res_dict["meas_noise_robustness"]["None"]["following_costs"], axis=1)
        #     norm_cost_follow = y2[0]
    else:
        norm_cost_free = 1.0
        norm_cost_follow = 1.0

    crits = []

    # model robustness
    if "model_robustness" in res_dict.keys():
        crits.append("model")
        mpar = model_parameters()
        mpar.load_yaml(os.path.join(results_dir, "model_parameters.yml"))
        mpar_dict = mpar.get_dict()

        fig_mr, ax_mr = plt.subplots(
            5, 2, figsize=(32 * scale, 24 * scale), num=fig_counter
        )
        fig_mr.suptitle("Model Robustness")
        for i, mp in enumerate(res_dict["model_robustness"].keys()):
            j = int(i % 5)
            k = int(i / 5)
            # ax[j][k].set_title(f"{mp}")

            ymax = 0.0

            x = res_dict["model_robustness"][mp]["values"]
            if "free_costs" in res_dict["model_robustness"][mp].keys():
                y1 = (
                    np.asarray(res_dict["model_robustness"][mp]["free_costs"])
                    / norm_cost_free
                )
                ax_mr[j][k].plot(x, y1, "o")
                ymax = np.max([ymax, np.max(y1)])
            # if "following_costs" in res_dict["model_robustness"][mp].keys():
            #     y2 = np.asarray(res_dict["model_robustness"][mp]["following_costs"]) / norm_cost_follow
            #     ax_mr[j][k].plot(x, y2, "o-")
            #     ymax = np.max([ymax, np.max(y2)])

            if costlim is not None:
                ymax = min(costlim[1], 1.1 * ymax)
            # ymax = np.max([np.max(y1), np.max(y2)])  # costlim[1]

            if "successes" in res_dict["model_robustness"][mp].keys():
                xr = x[:-1] + 0.5 * np.diff(x)
                xr = np.append([x[0]], xr)
                xr = np.append(xr, [x[-1]])
                succ = res_dict["model_robustness"][mp]["successes"]
                for ii in range(len(xr[:-1])):
                    c = "red"
                    if succ[ii]:
                        c = "green"
                    ax_mr[j][k].add_patch(
                        Rectangle(
                            (xr[ii], 0.0),
                            xr[ii + 1] - xr[ii],
                            ymax,
                            facecolor=c,
                            edgecolor=None,
                            alpha=0.1,
                        )
                    )
            if mp == "m1r1":
                temp = mpar_dict["m1"] * mpar_dict["r1"]
                mpar_x = [temp, temp]
            elif mp == "m2r2":
                temp = mpar_dict["m2"] * mpar_dict["r2"]
                mpar_x = [temp, temp]
            else:
                mpar_x = [mpar_dict[mp], mpar_dict[mp]]
            mpar_y = [0, ymax]
            ax_mr[j][k].plot(mpar_x, mpar_y, "--", color="grey")

            # if costlim is not None:
            #    ax_mr[j][k].set_ylim(costlim[0], costlim[1])
            if ymax > 2:
                ymin = 0.0
            else:
                ymin = 0.95
            ax_mr[j][k].set_ylim(ymin, ymax)
            ax_mr[j][k].set_ylabel("rel. cost")
            ax_mr[j][k].set_xlabel(mp)
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.5
        )
        if save:
            plt.savefig(
                os.path.join(results_dir, "model_robustness." + file_format),
                bbox_inches="tight",
            )
        fig_counter += 1

    # noise robustness
    if "meas_noise_robustness" in res_dict.keys():
        crits.append(r"$\dot{q}$ noise")
        n_subplots = len(res_dict["meas_noise_robustness"].keys())
        fig_nr, ax_nr = plt.subplots(
            n_subplots,
            1,
            figsize=(18 * scale, 12 * scale),
            sharex="all",
            num=fig_counter,
            squeeze=False,
        )
        fig_nr.suptitle("State Noise Robustness")
        for i, nf in enumerate(res_dict["meas_noise_robustness"].keys()):
            ymax = 0.0
            ax_nr[i][0].set_title(f"{nf}")
            x = res_dict["meas_noise_robustness"][nf]["noise_sigma_list"]
            if "free_costs" in res_dict["meas_noise_robustness"][nf].keys():
                y1 = (
                    np.median(
                        res_dict["meas_noise_robustness"][nf]["free_costs"], axis=1
                    )
                    / norm_cost_free
                )
                ax_nr[i][0].plot(x, y1, "o-")
                ymax = np.max([ymax, np.max(y1)])
            # if "following_costs" in res_dict["meas_noise_robustness"][nf].keys():
            #     y2 = np.median(res_dict["meas_noise_robustness"][nf]["following_costs"], axis=1) / norm_cost_follow
            #     ax_nr[i].plot(x, y2, "o-")
            #     ymax = np.max([ymax, np.max(y2)])

            if costlim is not None:
                ymax = min(costlim[1], ymax)

            if "successes" in res_dict["meas_noise_robustness"][nf].keys():
                xr = x[:-1] + 0.5 * np.diff(x)
                xr = np.append([x[0]], xr)
                xr = np.append(xr, [x[-1]])
                succs = res_dict["meas_noise_robustness"][nf]["successes"]
                succ = np.sum(succs, axis=1)
                for j in range(len(xr[:-1])):
                    c = "red"
                    if succ[j] > 0.5 * np.shape(succs)[-1]:
                        c = "green"
                    ax_nr[i][0].add_patch(
                        Rectangle(
                            (xr[j], 0.0),
                            xr[j + 1] - xr[j],
                            ymax,
                            facecolor=c,
                            edgecolor=None,
                            alpha=0.1,
                        )
                    )
            if ymax > 2:
                ymin = 0.0
            else:
                ymin = 0.95
            ax_nr[i][0].set_ylim(ymin, ymax)
            ax_nr[i][0].set_ylabel("rel. cost")
        ax_nr[-1][0].set_xlabel("Noise Variance")
        if save:
            plt.savefig(
                os.path.join(results_dir, "meas_noise_robustness." + file_format)
            )
        fig_counter += 1

    # unoise robustness
    if "u_noise_robustness" in res_dict.keys():
        crits.append(r"$\tau$ noise")
        ymax = 0.0
        # plt.figure(fig_counter, figsize=(16, 9))
        fig_unr, ax_unr = plt.subplots(
            1, 1, figsize=(16 * scale, 9 * scale), num=fig_counter
        )
        fig_unr.suptitle("Torque Noise Robustness")
        x = res_dict["u_noise_robustness"]["u_noise_sigma_list"]
        if "free_costs" in res_dict["u_noise_robustness"].keys():
            y1 = (
                np.median(res_dict["u_noise_robustness"]["free_costs"], axis=1)
                / norm_cost_free
            )
            ax_unr.plot(x, y1, "o-")
            ymax = np.max([ymax, np.max(y1)])
        # if "following_costs" in res_dict["u_noise_robustness"].keys():
        #     y2 = np.median(res_dict["u_noise_robustness"]["following_costs"], axis=1) / norm_cost_follow
        #     ax_unr.plot(x, y2, "o-")
        #     ymax = np.max([ymax, np.max(y2)])

        if costlim is not None:
            ymax = min(costlim[1], ymax)

        if "successes" in res_dict["u_noise_robustness"].keys():
            xr = x[:-1] + 0.5 * np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            succs = res_dict["u_noise_robustness"]["successes"]
            succ = np.sum(succs, axis=1)
            for i in range(len(xr[:-1])):
                c = "red"
                if succ[i] > 0.5 * np.shape(succs)[-1]:
                    c = "green"
                ax_unr.add_patch(
                    Rectangle(
                        (xr[i], 0.0),
                        xr[i + 1] - xr[i],
                        ymax,
                        facecolor=c,
                        edgecolor=None,
                        alpha=0.1,
                    )
                )
        if ymax > 2:
            ymin = 0.0
        else:
            ymin = 0.95
        ax_unr.set_ylim(ymin, ymax)
        ax_unr.set_xlabel("Noise Variance")
        ax_unr.set_ylabel("rel. cost")
        if save:
            plt.savefig(os.path.join(results_dir, "u_noise_robustness." + file_format))
        fig_counter += 1

    # u responsiveness robustness
    if "u_responsiveness_robustness" in res_dict.keys():
        crits.append(r"$\tau$ response")
        ymax = 0.0
        # plt.figure(fig_counter, figsize=(16, 9))
        fig_urr, ax_urr = plt.subplots(
            1, 1, figsize=(16 * scale, 9 * scale), num=fig_counter
        )
        fig_urr.suptitle("Motor Responsiveness Robustness")
        x = res_dict["u_responsiveness_robustness"]["u_responsivenesses"]
        if "free_costs" in res_dict["u_responsiveness_robustness"].keys():
            y1 = (
                np.asarray(res_dict["u_responsiveness_robustness"]["free_costs"])
                / norm_cost_free
            )
            ax_urr.plot(x, y1, "o-")
            ymax = np.max([ymax, np.max(y1)])
        # if "following_costs" in res_dict["u_responsiveness_robustness"].keys():
        #     y2 = np.asarray(res_dict["u_responsiveness_robustness"]["following_costs"]) / norm_cost_follow
        #     ax_urr.plot(x, y2, "o-")
        #     ymax = np.max([ymax, np.max(y2)])

        if costlim is not None:
            ymax = min(costlim[1], ymax)

        if "successes" in res_dict["u_responsiveness_robustness"].keys():
            xr = x[:-1] + 0.5 * np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            succ = res_dict["u_responsiveness_robustness"]["successes"]
            for i in range(len(xr[:-1])):
                c = "red"
                if succ[i]:
                    c = "green"
                ax_urr.add_patch(
                    Rectangle(
                        (xr[i], 0.0),
                        xr[i + 1] - xr[i],
                        ymax,
                        facecolor=c,
                        edgecolor=None,
                        alpha=0.1,
                    )
                )
        if ymax > 2:
            ymin = 0.0
        else:
            ymin = 0.95
        ax_urr.set_ylim(ymin, ymax)
        ax_urr.set_xlabel("Responsiveness Factor Amplitude")
        ax_urr.set_ylabel("rel. cost")
        if save:
            plt.savefig(os.path.join(results_dir, "u_responsivenesses." + file_format))
        fig_counter += 1

    # delay robustness
    if "delay_robustness" in res_dict.keys():
        crits.append("delay")
        ymax = 0.0
        fig_dr, ax_dr = plt.subplots(
            1, 1, figsize=(16 * scale, 9 * scale), num=fig_counter
        )
        # plt.figure(fig_counter, figsize=(16, 9))
        fig_dr.suptitle("Time Delay Robustness")
        x = res_dict["delay_robustness"]["measurement_delay"]
        if "free_costs" in res_dict["delay_robustness"].keys():
            y1 = np.asarray(res_dict["delay_robustness"]["free_costs"]) / norm_cost_free
            ax_dr.plot(x, y1, "o-")
            ymax = np.max([ymax, np.max(y1)])
        # if "following_costs" in res_dict["delay_robustness"].keys():
        #     y2 = np.asarray(res_dict["delay_robustness"]["following_costs"]) / norm_cost_follow
        #     ax_dr.plot(x, y2, "o-")
        #     ymax = np.max([ymax, np.max(y2)])

        if costlim is not None:
            ymax = min(costlim[1], ymax)

        if "successes" in res_dict["delay_robustness"].keys():
            xr = x[:-1] + 0.5 * np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            succ = res_dict["delay_robustness"]["successes"]
            for i in range(len(xr[:-1])):
                c = "red"
                if succ[i]:
                    c = "green"
                ax_dr.add_patch(
                    Rectangle(
                        (xr[i], 0.0),
                        xr[i + 1] - xr[i],
                        ymax,
                        facecolor=c,
                        edgecolor=None,
                        alpha=0.1,
                    )
                )
        if ymax > 2:
            ymin = 0.0
        else:
            ymin = 0.95
        ax_dr.set_ylim(ymin, ymax)
        ax_dr.set_xlabel("Time Delay [s]")
        ax_dr.set_ylabel("rel. cost")
        if save:
            plt.savefig(os.path.join(results_dir, "delay_robustness." + file_format))
        fig_counter += 1
    if "perturbation_robustness" in res_dict.keys():
        crits.append("pert.")
        fig_counter += 1

    # bar plot
    fig_bar, ax_bar = plt.subplots(
        1, 1, figsize=(10 * scale, 6 * scale), num=fig_counter
    )
    scores = get_scores(results_dir, filename)
    # crits = [
    #     "model",
    #     r"$\dot{q}$ noise",
    #     r"$\tau$ noise",
    #     r"$\tau$ response",
    #     "delay",
    #     "pert.",
    # ]
    # numbers = [
    #     scores["model"],
    #     scores["measurement_noise"],
    #     scores["u_noise"],
    #     scores["u_responsiveness"],
    #     scores["delay"],
    #     scores["perturbation"],
    # ]
    numbers = []
    for k in scores.keys():
        numbers.append(scores[k])
    bars = ax_bar.bar(crits, numbers)
    colors = ["red", "blue", "green", "purple", "orange", "magenta"]
    for i in range(fig_counter):
        bars[i].set_color(colors[i])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Robustness Criteria")
    ax_bar.set_ylabel("Success Score")
    if save:
        plt.savefig(
            os.path.join(results_dir, "score_plot." + file_format), bbox_inches="tight"
        )
    fig_counter += 1

    if show:
        plt.show()


def plot_model_robustness_multi(
    results_dirs, costlim, show=True, labels=[], save_dir=""
):
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

    fig_mr, ax_mr = plt.subplots(5, 2, figsize=(32, 24), num=0)
    fig_mr.suptitle("Model Robustness")

    handles = []
    ymax = np.zeros(10)

    for results_dir in results_dirs:
        # load data
        pickle_path = os.path.join(results_dir, "results.pkl")
        f = open(pickle_path, "rb")
        res_dict = pickle.load(f)
        f.close()

        # ToDo: solve better
        if "meas_noise_robustness" in res_dict.keys():
            if "free_costs" in res_dict["meas_noise_robustness"]["None"].keys():
                y1 = np.median(
                    res_dict["meas_noise_robustness"]["None"]["free_costs"], axis=1
                )
                norm_cost_free = y1[0]
        else:
            norm_cost_free = 1.0
            norm_cost_follow = 1.0

        mpar = model_parameters()
        mpar.load_yaml(os.path.join(results_dir, "model_parameters.yml"))
        mpar_dict = mpar.get_dict()

        for i, mp in enumerate(res_dict["model_robustness"].keys()):
            j = int(i % 5)
            k = int(i / 5)

            x = res_dict["model_robustness"][mp]["values"]
            if "free_costs" in res_dict["model_robustness"][mp].keys():
                y1 = (
                    np.asarray(res_dict["model_robustness"][mp]["free_costs"])
                    / norm_cost_free
                )
                (p,) = ax_mr[j][k].plot(x, y1, "o-")
                if i == 0:
                    handles.append(p)
                ymax[i] = np.max([ymax[i], np.max(y1)])

            if costlim is not None:
                ymax[i] = min(costlim[1], 1.1 * ymax[i])

            # if "successes" in res_dict["model_robustness"][mp].keys():
            #     xr = x[:-1] + 0.5*np.diff(x)
            #     xr = np.append([x[0]], xr)
            #     xr = np.append(xr, [x[-1]])
            #     succ = res_dict["model_robustness"][mp]["successes"]
            #     for ii in range(len(xr[:-1])):
            #         c = "red"
            #         if succ[ii]:
            #             c = "green"
            #         ax_mr[j][k].add_patch(
            #                 Rectangle((xr[ii], 0.),
            #                           xr[ii+1]-xr[ii], ymax[i],
            #                           facecolor=c, edgecolor=None,
            #                           alpha=0.1))
            if mp == "m1r1":
                temp = mpar_dict["m1"] * mpar_dict["r1"]
                mpar_x = [temp, temp]
            elif mp == "m2r2":
                temp = mpar_dict["m2"] * mpar_dict["r2"]
                mpar_x = [temp, temp]
            else:
                mpar_x = [mpar_dict[mp], mpar_dict[mp]]
            mpar_y = [0, ymax[i]]
            ax_mr[j][k].plot(mpar_x, mpar_y, "--", color="grey")

            if ymax[i] > 2:
                ymin = 0.0
            else:
                ymin = 0.95
            ax_mr[j][k].set_ylim(ymin, ymax[i])
            ax_mr[j][k].set_ylabel("rel. cost")
            ax_mr[j][k].set_xlabel(mp)

    ax_mr[-1][0].legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(1.2, -0.3),
        fancybox=False,
        shadow=False,
        ncol=3,
    )

    plt.savefig(os.path.join(save_dir, "model_robustness.pdf"), bbox_inches="tight")
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.8
    )
    if show:
        plt.show()
