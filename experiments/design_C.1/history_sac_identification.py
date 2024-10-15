import os
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.controller.history_sac import HistorySACController

from src.python.double_pendulum.controller.history_sac.history_sac_controller import IdentificationController

if __name__ == '__main__':

    f = 0.5
    a = 1
    joint = 0
    controller = IdentificationController(joint, f, a)
    controller.init()

    # run experiment
    run_experiment(
        controller=controller,
        dt=0.002,
        t_final=10.0,
        can_port="can0",
        motor_ids=[3, 1],
        tau_limit=[6, 6],
        motor_directions=[1.0, -1.0],
        save_dir=os.path.join(f"data/joint{joint}_f{f}_a{a}")
    )

