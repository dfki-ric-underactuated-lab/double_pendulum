import os
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def plot_benchmark_results(results_dir, costlim=[0, 1e6], show=False):

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

    # load data
    pickle_path = os.path.join(results_dir, "results.pkl")
    f = open(pickle_path, 'rb')
    res_dict = pickle.load(f)
    f.close()

    fig_counter = 0

    # model robustness
    if "model_robustness" in res_dict.keys():
        fig_mr, ax_mr = plt.subplots(5, 2, figsize=(24, 15),
                                     num=fig_counter)
        fig_mr.suptitle("Model Robustness")
        for i, mp in enumerate(res_dict["model_robustness"].keys()):
            j = int(i % 5)
            k = int(i / 5)
            # ax[j][k].set_title(f"{mp}")
            x = res_dict["model_robustness"][mp]["values"]
            if "free_costs" in res_dict["model_robustness"][mp].keys():
                y1 = res_dict["model_robustness"][mp]["free_costs"]
                ax_mr[j][k].plot(x, y1)
            if "following_costs" in res_dict["model_robustness"][mp].keys():
                y2 = res_dict["model_robustness"][mp]["following_costs"]
                ax_mr[j][k].plot(x, y2)
            if "successes" in res_dict["model_robustness"][mp].keys():
                xr = x[:-1] + 0.5*np.diff(x)
                xr = np.append([x[0]], xr)
                xr = np.append(xr, [x[-1]])
                ymax = costlim[1]
                succ = res_dict["model_robustness"][mp]["successes"]
                for i in range(len(xr[:-1])):
                    c = "red"
                    if succ[i]:
                        c = "green"
                    ax_mr[j][k].add_patch(
                            Rectangle((xr[i], 0.),
                                      xr[i+1]-xr[i], ymax,
                                      facecolor=c, edgecolor=None,
                                      alpha=0.1))

            ax_mr[j][k].set_ylim(costlim[0], costlim[1])
            ax_mr[j][k].set_ylabel("Cost")
            ax_mr[j][k].set_xlabel(mp)
        plt.savefig(os.path.join(results_dir, "model_robustness"))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.4)
        fig_counter += 1

    # noise robustness
    if "meas_noise_robustness" in res_dict.keys():
        n_subplots = len(res_dict["meas_noise_robustness"].keys())
        fig_nr, ax_nr = plt.subplots(
                n_subplots, 1,
                figsize=(18, 12), sharex="all", num=fig_counter)
        fig_nr.suptitle("State Noise Robustness")
        for i, nf in enumerate(res_dict["meas_noise_robustness"].keys()):
            ax_nr[i].set_title(f"{nf}")
            x = res_dict["meas_noise_robustness"][nf]["noise_sigma_list"]
            if "free_costs" in res_dict["meas_noise_robustness"][nf].keys():
                y1 = np.median(res_dict["meas_noise_robustness"][nf]["free_costs"], axis=1)
                ax_nr[i].plot(x, y1)
            if "following_costs" in res_dict["meas_noise_robustness"][nf].keys():
                y2 = np.median(res_dict["meas_noise_robustness"][nf]["following_costs"], axis=1)
                ax_nr[i].plot(x, y2)
            if "successes" in res_dict["meas_noise_robustness"][nf].keys():
                xr = x[:-1] + 0.5*np.diff(x)
                xr = np.append([x[0]], xr)
                xr = np.append(xr, [x[-1]])
                ymax = costlim[1]
                succs = res_dict["meas_noise_robustness"][nf]["successes"]
                succ = np.sum(succs, axis=1)
                for j in range(len(xr[:-1])):
                    c = "red"
                    if succ[j] > 0.5*np.shape(succs)[-1]:
                        c = "green"
                    ax_nr[i].add_patch(
                            Rectangle((xr[j], 0.),
                                      xr[j+1]-xr[j], ymax,
                                      facecolor=c, edgecolor=None,
                                      alpha=0.1))
            ax_nr[i].set_ylim(costlim[0], costlim[1])
            ax_nr[i].set_ylabel("Cost")
        ax_nr[-1].set_xlabel("Noise Variance")
        plt.savefig(os.path.join(results_dir, "meas_noise_robustness"))
        fig_counter += 1

    # unoise robustness
    if "u_noise_robustness" in res_dict.keys():
        # plt.figure(fig_counter, figsize=(16, 9))
        fig_unr, ax_unr = plt.subplots(1, 1, figsize=(16, 9), num=fig_counter)
        fig_unr.suptitle("Torque Noise Robustness")
        x = res_dict["u_noise_robustness"]["u_noise_sigma_list"]
        if "following_costs" in res_dict["u_noise_robustness"].keys():
            y1 = np.median(res_dict["u_noise_robustness"]["following_costs"], axis=1)
            ax_unr.plot(x, y1)
        if "free_costs" in res_dict["u_noise_robustness"].keys():
            y2 = np.median(res_dict["u_noise_robustness"]["free_costs"], axis=1)
            ax_unr.plot(x, y2)
        if "successes" in res_dict["u_noise_robustness"].keys():
            xr = x[:-1] + 0.5*np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            ymax = costlim[1]
            succs = res_dict["u_noise_robustness"]["successes"]
            succ = np.sum(succs, axis=1)
            for i in range(len(xr[:-1])):
                c = "red"
                if succ[i] > 0.5*np.shape(succs)[-1]:
                    c = "green"
                ax_unr.add_patch(
                        Rectangle((xr[i], 0.),
                                  xr[i+1]-xr[i], ymax,
                                  facecolor=c, edgecolor=None,
                                  alpha=0.1))
        ax_unr.set_ylim(costlim[0], costlim[1])
        ax_unr.set_xlabel("Noise Variance")
        ax_unr.set_ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "u_noise_robustness"))
        fig_counter += 1

    # u responsiveness robustness
    if "u_responsiveness_robustness" in res_dict.keys():
        # plt.figure(fig_counter, figsize=(16, 9))
        fig_urr, ax_urr = plt.subplots(1, 1, figsize=(16, 9), num=fig_counter)
        fig_urr.suptitle("Motor Responsiveness Robustness")
        x = res_dict["u_responsiveness_robustness"]["u_responsivenesses"]
        if "following_costs" in res_dict["u_responsiveness_robustness"].keys():
            y1 = res_dict["u_responsiveness_robustness"]["following_costs"]
            ax_urr.plot(x, y1)
        if "free_costs" in res_dict["u_responsiveness_robustness"].keys():
            y2 = res_dict["u_responsiveness_robustness"]["free_costs"]
            ax_urr.plot(x, y2)
        if "successes" in res_dict["u_responsiveness_robustness"].keys():
            xr = x[:-1] + 0.5*np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            ymax = costlim[1]
            succ = res_dict["u_responsiveness_robustness"]["successes"]
            for i in range(len(xr[:-1])):
                c = "red"
                if succ[i]:
                    c = "green"
                ax_urr.add_patch(
                        Rectangle((xr[i], 0.),
                                  xr[i+1]-xr[i], ymax,
                                  facecolor=c, edgecolor=None,
                                  alpha=0.1))
        ax_urr.set_ylim(costlim[0], costlim[1])
        ax_urr.set_xlabel("Responsiveness Factor Amplitude")
        ax_urr.set_ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "u_responsivenesses"))
        fig_counter += 1

    # delay robustness
    if "delay_robustness" in res_dict.keys():
        fig_dr, ax_dr = plt.subplots(1, 1, figsize=(16, 9), num=fig_counter)
        # plt.figure(fig_counter, figsize=(16, 9))
        fig_dr.suptitle("Time Delay Robustness")
        x = res_dict["delay_robustness"]["measurement_delay"]
        if "following_costs" in res_dict["delay_robustness"].keys():
            y1 = res_dict["delay_robustness"]["following_costs"]
            ax_dr.plot(x, y1)
        if "free_costs" in res_dict["delay_robustness"].keys():
            y2 = res_dict["delay_robustness"]["free_costs"]
            ax_dr.plot(x, y2)
        if "successes" in res_dict["delay_robustness"].keys():
            xr = x[:-1] + 0.5*np.diff(x)
            xr = np.append([x[0]], xr)
            xr = np.append(xr, [x[-1]])
            ymax = costlim[1]
            succ = res_dict["delay_robustness"]["successes"]
            for i in range(len(xr[:-1])):
                c = "red"
                if succ[i]:
                    c = "green"
                ax_dr.add_patch(
                        Rectangle((xr[i], 0.),
                                  xr[i+1]-xr[i], ymax,
                                  facecolor=c, edgecolor=None,
                                  alpha=0.1))
        ax_dr.set_ylim(costlim[0], costlim[1])
        ax_dr.set_xlabel("Time Delay [s]")
        ax_dr.set_ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "delay_robustness"))
        fig_counter += 1

    if show:
        plt.show()
