Real System Leaderboard
=======================

The real system leaderboard compares the performance of different control
methods on the real hardware. The task for the controller is to swingup and balance
the acrobot/pendubot and keep the end-effector above the threshhold line.

The scripts for the leaderboard calculation can be found in
`leaderboard/simulation/acrobot <https://github.com/dfki-ric-underactuated-lab/double_pendulum/tree/main/leaderboard/real_hardware/acrobot>`__ .
for the acrobot and in
`leaderboard/simulation/pendubot <https://github.com/dfki-ric-underactuated-lab/double_pendulum/tree/main/leaderboard/real_hardware/pendubot>`__ .
for the pendubot.

Creating the Leaderboard
------------------------

For creating the leaderboard locally, you can run::

  python create_leaderboard.py --data-dir <data-dir> --save_to <save_to>

This script will compute the leaderboard scores and save to a csv file.

The structure of the `data-dir` should be as follows

  - data-dir
      - controller_1
          - experiment01
            - trajectory.csv
            - timeseries.png
          - experiment02
          - ...
          - experiment10
          - username.txt
          - short_description.txt
          - README.md
          - name.txt
          - video.gif
      - controller_2
        - ...

The files are:

    - `trajectory.csv` data recorded from the real system
    - `timeseries.png` plot of `trajectory.csv`
    - `name.txt` contains the name of the controller
    - `username.txt` contains the (github) username of the creator f the controller
    - `short_description.txt` contains a short description of the controller (1-2 sentences)
    - `README.md` contains a longer description of the controller
    - `video.gif` is a video of one of the 10 experiments

See
`here <https://github.com/dfki-ric-underactuated-lab/real_ai_gym_leaderboard/tree/main/data/acrobot/real_system>`__
for an example structure.

The leaderboard will be stored in the `save_to` path 
Additionally, the script will save individual scores in the
experiment folders of the `data-dir`.


Evaluating your own controller
------------------------------

.. note::

   For implementing your own controller see `here
   <https://dfki-ric-underactuated-lab.github.io/double_pendulum/software_structure.controller.html>`__

If you want to evaluate your own controller and compare it to the listed
controllers on the leaderboard, we offer remote experiments with the real hardware setup.
Please contact
shivesh.kumar@dfki.de, felix.wiebe@dfki.de or shubham.vyas@dfki.de for details
and scheduling.
We recommend submitting the controller first to the acrobot `simulation
leaderboard <https://dfki-ric-underactuated-lab.github.io/real_ai_gym_leaderboard/acrobot_simulation_performance_leaderboard.html>`__
and the `robustness
leaderboard <https://dfki-ric-underactuated-lab.github.io/real_ai_gym_leaderboard/acrobot_simulation_robustness_leaderboard.html>`__).

