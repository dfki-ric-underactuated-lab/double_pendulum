import os
import pickle
import numpy as np
import matplotlib as mpl
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
        pass

    # noise robustness
    if "noise_robustness" in res_dict.keys():
        for nm in res_dict["noise_robustness"].keys():
            x = res_dict["noise_robustness"][nm]["noise_amplitudes"]
            y1 = np.mean(res_dict["noise_robustness"][nm]["following_costs"], axis=1)
            # mode = res_dict['noise_robustness'][nm]['noise_mode']
            # y2 = res_dict["noise_robustness"]["free_costs"]
            plt.figure(fig_counter, figsize=(16, 9))
            plt.title(f"State Noise Robustness ({nm})")
            plt.plot(x, y1)
            # plt.plot(x, y2)
            plt.ylim(costlim[0], costlim[1])
            plt.xlabel("Noise Amplitude")
            plt.ylabel("Cost")
            plt.savefig(os.path.join(results_dir, "noise_robustness_"+nm))
            fig_counter += 1

    # unoise robustness
    if "unoise_robustness" in res_dict.keys():
        x = res_dict["unoise_robustness"]["unoise_amplitudes"]
        y = np.mean(res_dict["unoise_robustness"]["following_costs"], axis=1)
        plt.figure(fig_counter, figsize=(16, 9))
        plt.title("Torque Noise Robustness")
        plt.plot(x, y)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Noise Amplitude")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "unoise_robustness"))
        fig_counter += 1

    # u responsiveness robustness
    if "u_responsiveness_robustness" in res_dict.keys():
        x = res_dict["u_responsiveness_robustness"]["u_responsivenesses"]
        y = res_dict["u_responsiveness_robustness"]["following_costs"]
        plt.figure(fig_counter, figsize=(16, 9))
        plt.title("Motor Responsiveness Robustness")
        plt.plot(x, y)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Responsiveness Factor Amplitude")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "u_responsivenesses"))
        fig_counter += 1

    # delay robustness
    if "delay_robustness" in res_dict.keys():
        x = res_dict["delay_robustness"]["measurement_delay"]
        y = res_dict["delay_robustness"]["following_costs"]
        plt.figure(fig_counter, figsize=(16, 9))
        plt.title("Time Delay Robustness")
        plt.plot(x, y)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Time Delay [s]")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "delay_robustness"))
        fig_counter += 1

    if show:
        plt.show()
