Simulation Leaderboard
======================

The simulation leaderboard compares the performance of different control
methods in simulation. The task for the controller is to swingup and balance
the acrobot/pendubot and keep the end-effector above the threshhold line.

Creating the Leaderboard
------------------------

For creating the leaderboard locally, you can run::

  python create_leaderboard.py

This script will:

    1. Check for controllers in all files starting with 'con_'
    2. Check if the simulation data for these controllers already exists
    3. Generate the data if it does not exist
    4. Compute the leaderboard scores and save to a csv file


Leaderboard Parameters
----------------------

The leaderboard uses a fixed model of the double pendulum and fixed simulation parameters.
The parameters can be found in the `sim_parameters.py` file.

Evaluating your own controller
------------------------------

.. note::

   For implementing your own controller see `here
   <https://dfki-ric-underactuated-lab.github.io/double_pendulum/software_structure.controller.html>`__

If you want to evaluate your own controller and compare it to the listed
controllers on the leaderboard, you have to create a file with the name
`con_controllername.py`, where `controllername` should be the name of the method
your controller uses.

In that file you should create an instance of your controller with the name
`controller` (will be imported under this name from the other scripts).
Additionally, yout `con_controllername.py` file should contain a dictionary::

  leaderboard_config = {"csv_path": name + "/sim_swingup.csv",
                        "name": name,
                        "simple_name": "simple name",
                        "short_description": "Short controller description (max 100 characters)",
                        "readme_path": f"readmes/{name}.md",
                        "username": username}

where `name` is the `controllername` and `username` is your github username.
Please add a simple name to display in the leaderboard and a short description
of your controller with max. 100 characters. Do not use commas (,) in your
description as they are used as separators in the data table!
For participating on the official leaderboard, i.e. when you create a pull
request, please also add a readme markdown file which describes your controller
in `readmes/controllername.md`. Add references if applicable.

Feel free to import the model and simulation parameters from
`sim_parameters.py` if you need them to instantiate your controller.

To calculate the leaderboard scores for your controller do::

    python calculate_leaderboard_score.py con_controllername.py

This will simulate the double pendulum controlled by your controller and save
all relevant data in `data/controllername` along with the leaderboard scores.
