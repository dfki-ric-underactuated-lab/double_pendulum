import os
import pickle
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

    # model robustness
    if "model_robustness" in res_dict.keys():
        pass

    # noise robustness
    if "noise_robustness" in res_dict.keys():
        x = res_dict["noise_robustness"]["noise_amplitudes"]
        y1 = res_dict["noise_robustness"]["following_costs"]
        mode = res_dict['noise_robustness']['noise_mode']
        # y2 = res_dict["noise_robustness"]["free_costs"]
        plt.figure(2, figsize=(16, 9))
        plt.title(f"State Noise Robustness ({mode})")
        plt.plot(x, y1)
        # plt.plot(x, y2)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Noise Amplitude")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "noise_robustness"))

    # unoise robustness
    if "unoise_robustness" in res_dict.keys():
        x = res_dict["unoise_robustness"]["unoise_amplitudes"]
        y = res_dict["unoise_robustness"]["following_costs"]
        plt.figure(3, figsize=(16, 9))
        plt.title("Torque Noise Robustness")
        plt.plot(x, y)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Noise Amplitude")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "unoise_robustness"))

    # u responsiveness robustness
    if "u_responsiveness_robustness" in res_dict.keys():
        x = res_dict["u_responsiveness_robustness"]["u_responsivenesses"]
        y = res_dict["u_responsiveness_robustness"]["following_costs"]
        plt.figure(4, figsize=(16, 9))
        plt.title("Motor Responsiveness Robustness")
        plt.plot(x, y)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Responsiveness Factor Amplitude")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "u_responsivenesses"))

    # delay robustness
    if "delay_robustness" in res_dict.keys():
        x = res_dict["delay_robustness"]["measurement_delay"]
        y = res_dict["delay_robustness"]["following_costs"]
        plt.figure(5, figsize=(16, 9))
        plt.title("Time Delay Robustness")
        plt.plot(x, y)
        plt.ylim(costlim[0], costlim[1])
        plt.xlabel("Time Delay [s]")
        plt.ylabel("Cost")
        plt.savefig(os.path.join(results_dir, "delay_robustness"))

    if show:
        plt.show()
