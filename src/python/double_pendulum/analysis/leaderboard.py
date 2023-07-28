import os
import numpy as np

from double_pendulum.utils.csv_trajectory import load_trajectory_full
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum


def leaderboard_scores(
    data_paths,
    save_to,
    mpar,
    weights={
        "swingup_time": 0.2,
        "max_tau": 0.1,
        "energy": 0.0,
        "integ_tau": 0.1,
        "tau_cost": 0.0,
        "tau_smoothness": 0.6,
    },
    normalize={
        "swingup_time": 10.0,
        "max_tau": 1.0,
        "energy": 1.0,
        "integ_tau": 10.0,
        "tau_cost": 10.0,
        "tau_smoothness": 1.0,
    },
    link_base="",
    simulation=True,
):
    """leaderboard_scores.
    Compute leaderboard scores from data_dictionaries which will be loaded from
    data_paths.  Data can be either from simulation or experiments (but for
    comparability it should only be one).

    Parameters
    ----------
    data_paths : dict
        contains the names and paths to the trajectory data in the form:
        {controller1_name: {"csv_path": data_path1, "name": controller1_name, "username": username1},
         controller2_name: {"csv_path": data_path2, "name": controller2_name, "username": username2},
         ...}
    save_to : string
        path where the result will be saved as a csv file
    weights : dict
        dictionary containing the weights for the different criteria in the
        form:
        {"swingup_time": weight1,
         "max_tau": weight2,
         "energy": weight3,
         "integ_tau": weight4,
         "tau_cost": weight5,
         "tau_smoothness": weight6,
         "velocity_cost" : weight7}
         The weights should sum up to 1 for the final score to be in the range
         [0, 1].
    normalize : dict
        dictionary containing normalization constants for the different
        criteria in the form:
        {"swingup_time": norm1,
         "max_tau": norm2,
         "energy": norm3,
         "integ_tau": norm4,
         "tau_cost": norm5,
         "tau_smoothness": norm6,
         "velocity_cost": norm7}
         The normalization constants should be the maximum values that can be
         achieved by the criteria so that after dividing by the norm the result
         is in the range [0, 1].
    simulation : bool
        whether to load the simulaition trajectory data
    link_base : string
        base-link for hosting data. Not needed for local execution
    """

    leaderboard_data = []

    for key in data_paths:
        d = data_paths[key]
        if type(d["csv_path"]) == str:
            csv_paths = [d["csv_path"]]
        else:
            csv_paths = d["csv_path"]

        swingup_times = []
        max_taus = []
        energies = []
        integ_taus = []
        tau_costs = []
        tau_smoothnesses = []
        velocity_costs = []
        successes = []
        scores = []

        for path in sorted(csv_paths):
            data_dict = load_trajectory_full(path)
            T = data_dict["T"]
            X = data_dict["X_meas"]
            U = data_dict["U_con"]

            swingup_times.append(
                get_swingup_time(
                    T=T, X=X, has_to_stay=True, mpar=mpar, method="height", height=0.9
                )
            )
            max_taus.append(get_max_tau(U))
            energies.append(get_energy(X, U))
            integ_taus.append(get_integrated_torque(T, U))
            tau_costs.append(get_torque_cost(T, U))
            tau_smoothnesses.append(get_tau_smoothness(U))
            velocity_costs.append(get_velocity_cost(T, X))

            successes.append(int(swingup_times[-1] < T[-1]))

            score = successes[-1] * (
                1.0
                - (
                    weights["swingup_time"]
                    * swingup_times[-1]
                    / normalize["swingup_time"]
                    + weights["max_tau"] * max_taus[-1] / normalize["max_tau"]
                    + weights["energy"] * energies[-1] / normalize["energy"]
                    + weights["integ_tau"] * integ_taus[-1] / normalize["integ_tau"]
                    + weights["tau_cost"] * tau_costs[-1] / normalize["tau_cost"]
                    + weights["tau_smoothness"]
                    * tau_smoothnesses[-1]
                    / normalize["tau_smoothness"]
                    + weights["velocity_cost"]
                    * velocity_costs[-1]
                    / normalize["velocity_cost"]
                )
            )

            scores.append(score)

            results = np.array(
                [
                    [successes[-1]],
                    [swingup_times[-1]],
                    [energies[-1]],
                    [max_taus[-1]],
                    [integ_taus[-1]],
                    [tau_costs[-1]],
                    [tau_smoothnesses[-1]],
                    [velocity_costs[-1]],
                    [score],
                ]
            ).T

            np.savetxt(
                os.path.join(os.path.dirname(path), "scores.csv"),
                results,
                header="Swingup Success,Swingup Time [s],Energy [J],Max. Torque [Nm],Integrated Torque [Nms],Torque Cost[N²m²],Torque Smoothness [Nm],Velocity Cost [m²/s²],RealAI Score",
                delimiter=",",
                fmt="%s",
                comments="",
            )

        best = np.argmax(scores)
        swingup_time = swingup_times[best]
        max_tau = max_taus[best]
        energy = energies[best]
        integ_tau = integ_taus[best]
        tau_cost = tau_costs[best]
        tau_smoothness = tau_smoothnesses[best]
        velocity_cost = velocity_costs[best]
        success = np.sum(successes)
        score = np.mean(scores)
        best_score = np.max(scores)

        if link_base != "":
            if "simple_name" in d.keys():
                name_with_link = (
                    f"[{d['simple_name']}]({link_base}{d['name']}/README.md)"
                )
            else:
                name_with_link = f"[{d['name']}]({link_base}{d['name']}/README.md)"
        else:
            if "simple_name" in d.keys():
                name_with_link = d["simple_name"]
            else:
                name_with_link = d["name"]

        if simulation:
            append_data = [
                name_with_link,
                d["short_description"],
                str(int(success)) + "/" + str(len(csv_paths)),
                str(round(swingup_time, 2)),
                str(round(energy, 2)),
                str(round(max_tau, 2)),
                str(round(integ_tau, 2)),
                str(round(tau_cost, 2)),
                str(round(tau_smoothness, 3)),
                str(round(velocity_cost, 2)),
                str(round(score, 3)),
                d["username"],
            ]
        else:
            append_data = [
                name_with_link,
                d["short_description"],
                str(int(success)) + "/" + str(len(csv_paths)),
                str(round(swingup_time, 2)),
                str(round(energy, 2)),
                str(round(max_tau, 2)),
                str(round(integ_tau, 2)),
                str(round(tau_cost, 2)),
                str(round(tau_smoothness, 3)),
                str(round(velocity_cost, 2)),
                str(round(best_score, 3)),
                str(round(score, 3)),
                d["username"],
            ]

        if link_base != "":
            controller_link = link_base + d["name"]

            if simulation:
                data_link = "[data](" + controller_link + "/sim_swingup.csv)"
                plot_link = "[plot](" + controller_link + "/timeseries.png)"
                video_link = "[video](" + controller_link + "/sim_video.gif)"
                append_data.append(data_link + " " + plot_link + " " + video_link)
            else:
                data_link = (
                    "[data]("
                    + controller_link
                    + "/experiment"
                    + str(best + 1).zfill(2)
                    + "/trajectory.csv)"
                )
                plot_link = (
                    "[plot]("
                    + controller_link
                    + "/experiment"
                    + str(best + 1).zfill(2)
                    + "/timeseries.png)"
                )
                video_link = (
                    "[video]("
                    + controller_link
                    + "/experiment"
                    + str(best + 1).zfill(2)
                    + "/video.gif)"
                )
                # link = "[data plots videos](" + controller_link + ")"
                # append_data.append(link)
                append_data.append(data_link + " " + plot_link + " " + video_link)

        leaderboard_data.append(append_data)

    if simulation:
        header = "Controller,Short Controller Description,Swingup Success,Swingup Time [s],Energy [J],Max. Torque [Nm],Integrated Torque [Nms],Torque Cost[N²m²],Torque Smoothness [Nm],Velocity Cost [m²/s²],RealAI Score,Username"
    else:
        header = "Controller,Short Controller Description,Swingup Success,Swingup Time [s],Energy [J],Max. Torque [Nm],Integrated Torque [Nms],Torque Cost[N²m²],Torque Smoothness [Nm],Velocity Cost [m²/s²],Best RealAI Score,Average RealAI Score,Username"
    if link_base != "":
        header += ",Data"

    np.savetxt(
        save_to,
        leaderboard_data,
        header=header,
        delimiter=",",
        fmt="%s",
        comments="",
    )


def get_swingup_time(
    T,
    X,
    eps=[1e-2, 1e-2, 1e-2, 1e-2],
    has_to_stay=True,
    mpar=None,
    method="height",
    height=0.9,
):
    """get_swingup_time.
    get the swingup time from a data_dict.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]
    eps : list
        list with len(eps) = 4. The thresholds for the swingup to be
        successfull ([position, velocity])
        default = [1e-2, 1e-2, 1e-2, 1e-2]
    has_to_stay : bool
        whether the pendulum has to stay upright until the end of the trajectory
        default=True

    Returns
    -------
    float
        swingup time
    """
    if method == "epsilon":
        goal = np.array([np.pi, 0.0, 0.0, 0.0])

        dist_x0 = np.abs(np.mod(X.T[0] - goal[0] + np.pi, 2 * np.pi) - np.pi)
        ddist_x0 = np.where(dist_x0 < eps[0], 0.0, dist_x0)
        n_x0 = np.argwhere(ddist_x0 == 0.0)

        dist_x1 = np.abs(np.mod(X.T[1] - goal[1] + np.pi, 2 * np.pi) - np.pi)
        ddist_x1 = np.where(dist_x1 < eps[1], 0.0, dist_x1)
        n_x1 = np.argwhere(ddist_x1 == 0.0)

        dist_x2 = np.abs(X.T[2] - goal[2])
        ddist_x2 = np.where(dist_x2 < eps[2], 0.0, dist_x2)
        n_x2 = np.argwhere(ddist_x2 == 0.0)

        dist_x3 = np.abs(X.T[3] - goal[3])
        ddist_x3 = np.where(dist_x3 < eps[3], 0.0, dist_x3)
        n_x3 = np.argwhere(ddist_x3 == 0.0)

        n = np.intersect1d(n_x0, n_x1)
        n = np.intersect1d(n, n_x2)
        n = np.intersect1d(n, n_x3)

        time_index = len(T) - 1
        if has_to_stay:
            if len(n) > 0:
                for i in range(len(n) - 2, 0, -1):
                    if n[i] + 1 == n[i + 1]:
                        time_index = n[i]
                    else:
                        break
        else:
            if len(n) > 0:
                time_index = n[0]
        time = T[time_index]
    elif method == "height":
        plant = SymbolicDoublePendulum(model_pars=mpar)
        fk = plant.forward_kinematics(X.T[:2])
        ee_pos_y = fk[1][1]

        goal_height = height * (mpar.l[0] + mpar.l[1])

        up = np.where(ee_pos_y > goal_height, True, False)

        time_index = len(T) - 1
        if has_to_stay:
            for i in range(len(up) - 2, 0, -1):
                if up[i]:
                    time_index = i
                else:
                    break

        else:
            time_index = np.argwhere(up)[0][0]
        time = T[time_index]

    else:
        time = np.inf

    return time


def get_max_tau(U):
    """get_max_tau.

    Get the maximum torque used in the trajectory.

    Parameters
    ----------
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        maximum torque
    """
    tau = np.max(np.abs(U))
    return tau


def get_energy(X, U):
    """get_energy.

    Get the mechanical energy used during the swingup.

    Parameters
    ----------
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        energy
    """

    delta_pos0 = np.diff(X.T[0])
    tau0 = U.T[0][:-1]
    energy0 = np.sum(np.abs(delta_pos0 * tau0))

    delta_pos1 = np.diff(X.T[1])
    tau1 = U.T[1][:-1]
    energy1 = np.sum(np.abs(delta_pos1 * tau1))

    energy = energy0 + energy1

    return energy


def get_integrated_torque(T, U):
    """get_integrated_torque.

    Get the (discrete) time integral over the torque.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        integrated torque
    """
    delta_t = np.diff(T)
    tau0 = np.abs(U.T[0][:-1])
    int_tau0 = np.sum(tau0 * delta_t)

    tau1 = np.abs(U.T[1][:-1])
    int_tau1 = np.sum(tau1 * delta_t)

    int_tau = int_tau0 + int_tau1

    return int_tau


def get_torque_cost(T, U, R=np.diag([1.0, 1.0])):
    """get_torque_cost.

    Get the running cost torque with cost parameter R.
    The cost is normalized with the timestep.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]
    R : numpy array
        running cost weight matrix (2x2)

    Returns
    -------
    float
        torque cost
    """
    delta_t = np.diff(T)
    cost = np.einsum("ij, i, jk, ik", U[:-1], delta_t, R, U[:-1])
    return cost


def get_tau_smoothness(U):
    """get_tau_smoothness.

    Get the standard deviation of the changes in the torque signal.

    Parameters
    ----------
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]

    Returns
    -------
    float
        torque smoothness (std of changes)
    """
    u_diff0 = np.diff(U.T[0])
    std0 = np.std(u_diff0)

    u_diff1 = np.diff(U.T[1])
    std1 = np.std(u_diff1)

    std = std0 + std1

    return std


def get_velocity_cost(T, X, Q=np.diag([1.0, 1.0])):
    """get_torque_cost.

    Get the running velocity cost with cost matrix Q.
    The cost is normalized with the timestep.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    Q : numpy array
        running cost weight matrix (2x2)

    Returns
    -------
    float
        torque cost
    """
    delta_t = np.diff(T)
    V = X.T[2:].T
    cost = np.einsum("ij, i, jk, ik", V[:-1], delta_t, Q, V[:-1])
    return cost
