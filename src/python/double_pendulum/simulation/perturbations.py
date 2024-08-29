import numpy as np
import matplotlib.pyplot as plt


def get_gaussian_perturbation_array(
    tmax, dt, mu=[[], []], sigma=[[], []], amplitude=[[], []]
):
    t = np.arange(0, tmax, dt)

    gauss = [[], []]
    perturbation_array = []

    for j in range(2):
        for i in range(len(mu[j])):
            g = amplitude[j][i] * np.exp(
                -np.power(t - mu[j][i], 2.0) / (2 * sigma[j][i] ** 2.0)
            )
            gauss[j].append(g)
        if len(gauss[j]) > 0:
            perturbation_array.append(np.sum(gauss[j], axis=0))
        else:
            perturbation_array.append([])

    return np.asarray(perturbation_array)


def get_random_gauss_perturbation_array(
    tmax,
    dt,
    n_per_joint=3,
    min_t_dist=1.0,
    sigma_minmax=[0.01, 0.05],
    amplitude_min_max=[0.1, 1.0],
):
    n = 2 * n_per_joint
    wiggle_room = (tmax - (n + 1) * min_t_dist) / n
    mu = []
    for i in range(n):
        if i == 0:
            mu.append(min_t_dist + np.random.rand() * wiggle_room)
        else:
            mu.append(mu[-1] + min_t_dist + np.random.rand() * wiggle_room)
    np.random.shuffle(mu)
    mu = np.reshape(mu, (2, n_per_joint))

    sigma = sigma_minmax[0] + np.random.rand(2, n_per_joint) * (
        sigma_minmax[1] - sigma_minmax[0]
    )

    amp_range = amplitude_min_max[1] - amplitude_min_max[0]
    amplitudes = np.random.uniform(-amp_range, amp_range, size=(2, n_per_joint))
    amplitudes += amplitude_min_max[0] * np.sign(amplitudes)

    return (
        get_gaussian_perturbation_array(tmax, dt, mu, sigma, amplitudes),
        mu,
        sigma,
        amplitudes,
    )


def plot_perturbation_array(
    tmax,
    dt,
    perturbation_array,
    save_to=None,
    show=True,
):
    # t = np.arange(0, tmax, dt)
    t = np.linspace(0, tmax, len(perturbation_array[0]))

    if len(perturbation_array[0]) > 0:
        plt.plot(t, perturbation_array[0], label="joint 1")
    else:
        plt.plot(t, np.zeros(len(t)), label="joint 1")

    if len(perturbation_array[1]) > 0:
        plt.plot(t, perturbation_array[1], label="joint 2")
    else:
        plt.plot(t, np.zeros(len(t)), label="joint 2")

    plt.legend(loc="best")

    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()


def get_perturbation_starts(perturbation_array):
    inds = []
    for i in range(perturbation_array.shape[1] - 1):
        if (
            np.max(np.abs(perturbation_array[:, i + 1])) >= 0.1
            and np.max(np.abs(perturbation_array[:, i])) < 0.1
        ):
            inds.append(i + 1)
    return inds
