import os
import numpy as np

from double_pendulum.utils.csv_trajectory import load_trajectory_full

# from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.plant import DoublePendulumPlant


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
        "velocity_cost": 0.2,
    },
    normalize={
        "swingup_time": 10.0,
        "max_tau": 1.0,
        "energy": 1.0,
        "integ_tau": 10.0,
        "tau_cost": 10.0,
        "tau_smoothness": 1.0,
        "velocity_cost": 1000,
    },
    link_base="",
    simulation=True,
    score_version="v1",
    t_final=10.0,
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
    score_version : string
        which equation should be used for the score calculation
        if set to something else than "v1", "v2" will be used
        default: "v1"
    """

    leaderboard_data = []

    all_criteria = [
        "swingup_time",
        "max_tau",
        "energy",
        "integ_tau",
        "tau_cost",
        "tau_smoothness",
        "velocity_cost",
        "uptime",
        "number_of_swingups",
    ]

    for crit in all_criteria:
        if crit not in weights.keys():
            weights[crit] = 0.0
        if crit not in normalize.keys():
            normalize[crit] = 1.0

    nonzero_weigths = 0
    for w in weights.keys():
        if weights[w] != 0.0:
            nonzero_weigths += 1

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
        n_swingups = []
        uptimes = []
        successes = []
        scores = []

        for path in sorted(csv_paths):
            data_dict = load_trajectory_full(path)
            T = data_dict["T"]
            X = data_dict["X_meas"]
            U = data_dict["U_con"]

            swingup_times.append(
                get_swingup_time(
                    T=T,
                    X=X,
                    has_to_stay=True,
                    mpar=mpar,
                    method="height",
                    height=0.9,
                    t_final=t_final,
                )
            )
            max_taus.append(get_max_tau(U))
            energies.append(get_energy(X, U))
            integ_taus.append(get_integrated_torque(T, U))
            tau_costs.append(get_torque_cost(T, U))
            tau_smoothnesses.append(get_tau_smoothness(U))
            velocity_costs.append(get_velocity_cost(T, X))
            n_swingups.append(
                get_number_of_swingups(T, X, mpar=mpar, method="height", height=0.9)
            )
            uptimes.append(get_uptime(T, X, mpar=mpar, method="height", height=0.9))

            successes.append(int(swingup_times[-1] < T[-1]))

            if score_version == "v1":
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
            elif score_version == "v2":
                score = successes[-1] * (
                    1.0
                    - 1.0
                    / nonzero_weigths
                    * (
                        np.tanh(
                            np.pi
                            * weights["swingup_time"]
                            * swingup_times[-1]
                            / normalize["swingup_time"]
                        )
                        + np.tanh(
                            np.pi
                            * weights["max_tau"]
                            * max_taus[-1]
                            / normalize["max_tau"]
                        )
                        + np.tanh(
                            np.pi
                            * weights["energy"]
                            * energies[-1]
                            / normalize["energy"]
                        )
                        + np.tanh(
                            np.pi
                            * weights["integ_tau"]
                            * integ_taus[-1]
                            / normalize["integ_tau"]
                        )
                        + np.tanh(
                            np.pi
                            * weights["tau_cost"]
                            * tau_costs[-1]
                            / normalize["tau_cost"]
                        )
                        + np.tanh(
                            np.pi
                            * weights["tau_smoothness"]
                            * tau_smoothnesses[-1]
                            / normalize["tau_smoothness"]
                        )
                        + np.tanh(
                            np.pi
                            * weights["velocity_cost"]
                            * velocity_costs[-1]
                            / normalize["velocity_cost"]
                        )
                    )
                )
            elif score_version == "v3":
                score = weights["uptime"] * uptimes[-1] / normalize["uptime"]
            else:
                score = 0.0

            scores.append(score)

            header = ""
            results = []
            if score_version in ["v1", "v2"]:
                results.append([successes[-1]])
                header += "Swingup Success,"
            if weights["swingup_time"] != 0.0:
                results.append([swingup_times[-1]])
                header += "Swingup Time [s],"
            if weights["energy"] != 0.0:
                results.append([energies[-1]])
                header += "Energy [J],"
            if weights["max_tau"] != 0.0:
                results.append([max_taus[-1]])
                header += "Max. Torque [Nm],"
            if weights["integ_tau"] != 0.0:
                results.append([integ_taus[-1]])
                header += "Integrated Torque [Nms],"
            if weights["tau_cost"] != 0.0:
                results.append([tau_costs[-1]])
                header += "Torque Cost[N²m²],"
            if weights["tau_smoothness"] != 0.0:
                results.append([tau_smoothnesses[-1]])
                header += "Torque Smoothness [Nm],"
            if weights["velocity_cost"] != 0.0:
                results.append([velocity_costs[-1]])
                header += "Velocity Cost [m²/s²],"
            if weights["uptime"] != 0.0:  # intentionally checking for uptime
                results.append([n_swingups[-1]])
                header += "#swingups,"
            if weights["uptime"] != 0.0:
                results.append([uptimes[-1]])
                header += "Uptime [s],"
            results.append([score])
            header += "RealAI Score"
            results = np.asarray(results).T

            np.savetxt(
                os.path.join(os.path.dirname(path), f"scores_{score_version}.csv"),
                results,
                header=header,
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
        uptime = uptimes[best]
        n_swingup = n_swingups[best]
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

        append_data = [name_with_link, d["short_description"]]
        if score_version in ["v1", "v2"]:
            append_data.append(str(int(success)) + "/" + str(len(csv_paths)))
        if weights["swingup_time"] != 0.0:
            append_data.append(str(round(swingup_time, 2)))
        if weights["energy"] != 0.0:
            append_data.append(str(round(energy, 2)))
        if weights["max_tau"] != 0.0:
            append_data.append(str(round(max_tau, 2)))
        if weights["integ_tau"] != 0.0:
            append_data.append(str(round(integ_tau, 2)))
        if weights["tau_cost"] != 0.0:
            append_data.append(str(round(tau_cost, 2)))
        if weights["tau_smoothness"] != 0.0:
            append_data.append(str(round(tau_smoothness, 3)))
        if weights["velocity_cost"] != 0.0:
            append_data.append(str(round(velocity_cost, 2)))
        if weights["uptime"] != 0.0:  # intentionally checking for uptime
            append_data.append(str(n_swingup))
        if weights["uptime"] != 0.0:
            append_data.append(str(round(uptime, 3)))

        if simulation:
            append_data.append(str(round(score, 3)))
            append_data.append(d["username"])

            if link_base != "":
                controller_link = link_base + d["name"]

                data_link = "[data](" + controller_link + "/sim_swingup.csv)"
                plot_link = "[plot](" + controller_link + "/timeseries.png)"
                video_link = "[video](" + controller_link + "/sim_video.gif)"
                append_data.append(data_link + " " + plot_link + " " + video_link)
        else:
            append_data.append(str(round(best_score, 3)))
            append_data.append(str(round(score, 3)))
            append_data.append(d["username"])
            if link_base != "":
                controller_link = link_base + d["name"]
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
                append_data.append(data_link + " " + plot_link + " " + video_link)

        leaderboard_data.append(append_data)

    header = "Controller,"
    header += "Short Controller Description,"
    if score_version in ["v1", "v2"]:
        header += "Swingup Success,"
    if weights["swingup_time"] != 0.0:
        header += "Swingup Time [s],"
    if weights["energy"] != 0.0:
        header += "Energy [J],"
    if weights["max_tau"] != 0.0:
        header += "Max. Torque [Nm],"
    if weights["integ_tau"] != 0.0:
        header += "Integrated Torque [Nms],"
    if weights["tau_cost"] != 0.0:
        header += "Torque Cost[N²m²],"
    if weights["tau_smoothness"] != 0.0:
        header += "Torque Smoothness [Nm],"
    if weights["velocity_cost"] != 0.0:
        header += "Velocity Cost [m²/s²],"
    if weights["uptime"] != 0.0:  # intentionally checking for uptime
        header += "#swingups,"
    if weights["uptime"] != 0.0:
        header += "Uptime [s],"

    if simulation:
        header += "RealAI Score,"
        header += "Username"
    else:
        header += "Best RealAI Score,"
        header += "Average RealAI Score,"
        header += "Username"

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
    t_final=10.0,
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
    if T[-1] < 0.99 * t_final:
        time = np.inf
    elif method == "epsilon":
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
        # plant = SymbolicDoublePendulum(model_pars=mpar)
        plant = DoublePendulumPlant(model_pars=mpar)
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


def check_if_up_epsilon(x, eps=[1e-2, 1e-2, 1e-2, 1e-2]):
    goal = np.array([np.pi, 0.0, 0.0, 0.0])

    X_error = np.abs(x - goal)
    up = np.all(np.where(X_error < eps, True, False), axis=1)
    return up


def check_if_up_height(x, height=0.9, mpar=None, eps=[1e-2, 1e-2, 1e-2, 1e-2]):
    plant = DoublePendulumPlant(model_pars=mpar)
    fk = plant.forward_kinematics(x[:2])
    ee_pos_y = fk[1][1]

    goal_height = height * (mpar.l[0] + mpar.l[1])

    up = ee_pos_y > goal_height  # and np.abs(x[2]) < eps[2] and np.abs(x[3]) < eps[3]
    return up


def check_if_up(
    x, method="height", mpar=None, eps=[1e-2, 1e-2, 1e-2, 1e-2], height=0.9
):
    if method == "epsilon":
        up = check_if_up_epsilon(x, eps)
    elif method == "height":
        up = check_if_up_height(x, height, mpar, eps)
    else:
        up = False
    return up


def get_uptime(
    T,
    X,
    eps=[1e-2, 1e-2, 1e-2, 1e-2],
    mpar=None,
    method="height",
    height=0.9,
):

    DT = np.diff(T, prepend=0.0)

    if method == "epsilon":
        goal = np.array([np.pi, 0.0, 0.0, 0.0])

        X_error = np.abs(X - goal)
        up = np.all(np.where(X_error < eps, True, False), axis=1)
        uptime = np.sum(DT[up])

    elif method == "height":
        plant = DoublePendulumPlant(model_pars=mpar)
        fk = plant.forward_kinematics(X.T[:2])
        ee_pos_y = fk[1][1]

        goal_height = height * (mpar.l[0] + mpar.l[1])

        up = np.where(ee_pos_y > goal_height, True, False)
        uptime = np.sum(DT[up])

    else:
        uptime = 0.0

    return uptime


def get_number_of_swingups(
    T,
    X,
    eps=[1e-2, 1e-2, 5e-1, 5e-1],
    mpar=None,
    method="height",
    height=0.9,
    deadtime=1.0,
):
    if method == "epsilon":
        goal = np.array([np.pi, 0.0, 0.0, 0.0])
        last_up_time = -deadtime
        last_step_up = False
        up = False
        n_swingups = 0
        for i, t in enumerate(T):
            up = check_if_up_epsilon(X[i], eps)

            if up and not last_step_up and (T[i] - last_up_time) > deadtime:
                n_swingups += 1
                last_up_time = T[i]

            last_step_up = up

    elif method == "height":
        last_up_time = -deadtime
        last_step_up = False
        up = False
        n_swingups = 0
        for i, t in enumerate(T):
            up = check_if_up_height(X[i], height, mpar, eps)

            if up and not last_step_up and (T[i] - last_up_time) > deadtime:
                n_swingups += 1
                last_up_time = T[i]

            last_step_up = np.copy(up)
    else:
        n_swingups = 0
    return n_swingups
