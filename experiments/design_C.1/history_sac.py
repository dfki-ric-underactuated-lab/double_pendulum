import os
from double_pendulum.experiments.hardware_control_loop_tmotors import run_experiment
from double_pendulum.controller.history_sac import HistorySACController


if __name__ == '__main__':

    env_type = "acrobot"
    model = "final2_only2"
    controller = HistorySACController(env_type, model_path="../../data/policies/design_C.1/model_1.1/" + env_type + "/history_sac/" + model)
    # controller.set_friction_compensation(damping=[0.001, 0.001], coulomb_fric=[0.16, 0.12])
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
        save_dir=os.path.join("data/" + env_type + "/data_tuesday2/" + model)
    )

