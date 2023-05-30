import os
import pickle
import numpy as np


def get_scores(results_dir, filename="results.pkl"):
    # load data
    pickle_path = os.path.join(results_dir, filename)
    f = open(pickle_path, "rb")
    res_dict = pickle.load(f)
    f.close()

    scores = {}
    scores["model"] = float(get_model_score(res_dict))
    scores["measurement_noise"] = float(get_measurement_noise_score(res_dict))
    scores["u_noise"] = float(get_unoise_score(res_dict))
    scores["u_responsiveness"] = float(get_uresponsiveness_score(res_dict))
    scores["delay"] = float(get_delay_score(res_dict))

    return scores


def get_model_score(res_dict):
    N = 0
    S = 0

    for i, mp in enumerate(res_dict["model_robustness"].keys()):
        succ = res_dict["model_robustness"][mp]["successes"]
        N += len(succ)
        S += np.sum(succ)
    return S / N


def get_measurement_noise_score(res_dict):
    N = 0
    S = 0

    for i, nf in enumerate(res_dict["meas_noise_robustness"].keys()):
        succs = res_dict["meas_noise_robustness"][nf]["successes"]
        succ = np.average(succs, axis=1)
        succ = np.where(succ > 0.5, True, False)
        N += len(succ)
        S += np.sum(succ)
    return S / N


def get_unoise_score(res_dict):
    succs = res_dict["u_noise_robustness"]["successes"]
    #succ = np.sum(succs, axis=1)
    succ = np.average(succs, axis=1)
    succ = np.where(succ > 0.5, True, False)
    N = len(succ)
    S = np.sum(succ)
    return S / N


def get_uresponsiveness_score(res_dict):
    succ = res_dict["u_responsiveness_robustness"]["successes"]
    N = len(succ)
    S = np.sum(succ)
    return S / N


def get_delay_score(res_dict):
    succ = res_dict["delay_robustness"]["successes"]
    N = len(succ)
    S = np.sum(succ)
    return S / N
