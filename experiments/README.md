# Hardware Experiments

This folder contains scripts to conduct experiments with the double pendulum hardware.
The scripts are sorted in subfolders for the respective designs.

Before performing experiments consider the following remarks as well as the
[hardware
documentation](https://dfki-ric-underactuated-lab.github.io/double_pendulum/hardware.html).
Especially, read the [safety
notes](https://dfki-ric-underactuated-lab.github.io/double_pendulum/hardware.experiments.html)
carefully.

## Can interface

The PC communicates with the motors via can. For this the can0 interface must
be brought up with

    . setup_caninterface.sh

You can check if the can interface it actually UP with

    ifconfig

It should show status UP. If there is a problem with enabling the motors it is
most likely to a interrupted can connection.

## mjbots

When running experiments with mjbots in a virtual python environment (e.g. double-pendulum38) you HAVE to run it with 

    sudo -E env PATH=$PATH

prepended to script call command

e.g. 

    sudo -E env PATH=$PATH python mjbots_acrobot_lqr.py

And also when zero-offsetting and stopping the motors inside the script with os.system("sudo -E env PATH=$PATH ...") this has to be prepended!
