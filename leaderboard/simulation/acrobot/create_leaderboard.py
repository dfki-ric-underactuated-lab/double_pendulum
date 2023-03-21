import os
import importlib
import pandas

from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.utils.csv_trajectory import save_trajectory
#from double_pendulum.analysis.leaderboard_utils import compute_leaderboard
from double_pendulum.analysis.leaderboard import leaderboard_scores

from sim_parameters import mpar, dt, t_final, t0, x0, goal, integrator


recompute_leaderboard = True

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

existing_list = os.listdir(data_dir)
for con in existing_list:
    if not os.path.exists(os.path.join("data", con, "sim_swingup.csv")):
        existing_list.remove(con)

for file in os.listdir("."):
    if file[:4] == "con_":
        if file[4:-3] in existing_list:
            print(f"Simulation data for {file} found.")
        else:
            print(f"Simulating new controller {file}")

            controller_arg = file[:-3]
            controller_name = controller_arg[4:]

            save_dir = f"data/{controller_name}"

            imp = importlib.import_module(controller_arg)

            controller = imp.controller

            plant = SymbolicDoublePendulum(model_pars=mpar)

            sim = Simulator(plant=plant)

            T, X, U = sim.simulate(t0=t0, x0=x0, tf=t_final, dt=dt, controller=controller, integrator=integrator)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_trajectory(os.path.join(save_dir, "sim_swingup.csv"), T=T, X_meas=X, U_con=U)

            recompute_leaderboard = True

if recompute_leaderboard:
    src_dir="."
    save_to="data/leaderboard.csv"
    data_paths = {}

    for f in os.listdir(src_dir):
        if f[:4] == "con_":
            mod = importlib.import_module(f[:-3])
            if hasattr(mod, "leaderboard_config"):
                if os.path.exists(mod.leaderboard_config["csv_path"]):
                    print(f"Found leaderboard_config and data for {mod.leaderboard_config['name']}")
                    data_paths[mod.leaderboard_config["name"]] = mod.leaderboard_config

    leaderboard_scores(
        data_paths,
        save_to,
        # weights={"swingup_time": 0.5, "max_tau": 0.1, "energy": 0.0, "integ_tau": 0.4, "tau_cost": 0.0, "tau_smoothness": 0.0},
        weights={
            "swingup_time": 0.4,
            "max_tau": 0.2,
            "energy": 0.0,
            "integ_tau": 0.2,
            "tau_cost": 0.0,
            "tau_smoothness": 0.2,
        },
        normalize={
            "swingup_time": 10.0,
            "max_tau": 6.0,
            "energy": 1.0,
            "integ_tau": 60.0,
            "tau_cost": 360.0,
            "tau_smoothness": 12.0,
        },
    )

    print(pandas.read_csv(save_to).sort_values(by=["Real AI Score"], ascending=False).to_markdown(index=False))
