Robustness Leaderboard
======================

The robustness leaderboard compares the performance of different control
methods by perturbing the simulation.
The task for the controller is to swingup and balance the acrobot/pendubot.

The scripts for the leaderboard calculation can be found in
leaderboard/robustness/acrobot (pendubot is coming soon).

Creating the Leaderboard
------------------------

For creating the leaderboard locally, you can run::

  python create_leaderboard.py

This script will:

    1. Check for controllers in all files starting with 'con_'
    2. Check if the robustness simulation data for these controllers already exists
    3. Generate the data if it does not exist
    4. Compute the leaderboard scores and save to a csv file


Leaderboard Parameters
----------------------

The leaderboard uses a fixed model of the double pendulum and fixed simulation parameters.
The parameters can be found in the `sim_parameters.py` file.
The perturbations applied during the simulation can be checked in the benchmark documentation.

Evaluating your own controller
------------------------------

If you want to evaluate your own controller and compare it to the listed
controllers on the leaderboard, you have to create a file with the name
`con_controllername.py`, where `controllername` should be the name of the method
your controller uses.
In that file you should create an instance of your controller with the name
`controller` (will be imported under this name from the other scripts).
Additionally, yout `con_controllername.py` file should contain a dictionary::

  leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                        "name": name,
                        "username": username}

where `name` is the `controllername` and `username` is your github username.

Feel free to import the model and simulation parameters from
`sim_parameters.py` if you need them to instantiate your controller.

You can now generate the robustness data by calling::

    python benchmark_controller.py con_controllername.py

This will simulate the double pendulum with all defined perturbations under the
control of your controlled and save all relevant data in `data/controllername`.

.. note::
  Depending on the speed of your controller and your PC this may take a while (1h-6h). 

If you create a pull request and your controller and the `con_controllername.py` 
is added to the main repository your controller will appear on the official leaderboard.
