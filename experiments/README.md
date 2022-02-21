When running experiments with mjbots in a virtual python environment (e.g. double-pendulum38) you HAVE to run it with 

sudo -E env PATH=$PATH

prepended to script call command

(e.g. sudo -E env PATH=$PATH python mjbots_acrobot_lqr.py).
And also when zero-offsetting and stopping the motors inside the script with os.system("sudo -E env PATH=$PATH ...") this has to be prepended!
