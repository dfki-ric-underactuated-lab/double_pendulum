
# Scripts for Recreating Paper Results

This folder contains all scripts that were used to produce the results that can
be found in the paper.

The scripts rely on data from the other scripts (e.g. the plotting
scripts rely on the data being computed first, etc).
Data and results are stored in a folder called "results" on the top level of
the repository ("../../results" from here).

To run all scripts in correct order you can use the "run" script via

    . ./run

The results of the computation can also be found in the "data" folder of this
repository. The figures from the paper are stored at "docs/figures/paper".

Note: A complete run can take >30h (of course depending on the computer).

Note: The scripts with names starting with 'experiment_' are the scripts used
in the hardware experiments. They are not included in the "run" script.

Note: Run this code with release
[0.1.1](https://github.com/dfki-ric-underactuated-lab/double_pendulum/releases/tag/v0.1.1).
