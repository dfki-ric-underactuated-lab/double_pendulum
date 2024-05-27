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

    return perturbation_array


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


def plot_perturbation_array(tmax, dt, perturbation_array):
    t = np.arange(0, tmax, dt)

    if len(perturbation_array[0]) > 0:
        plt.plot(t, perturbation_array[0], label="joint 1")
    else:
        plt.plot(t, np.zeros(len(t)), label="joint 1")

    if len(perturbation_array[1]) > 0:
        plt.plot(t, perturbation_array[1], label="joint 2")
    else:
        plt.plot(t, np.zeros(len(t)), label="joint 2")

    plt.legend(loc="best")
    plt.show()
