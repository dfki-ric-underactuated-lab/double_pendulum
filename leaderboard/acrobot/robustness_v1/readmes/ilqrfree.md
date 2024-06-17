# Iterative Linear Quadratic Regulator (iLQR) MPC without nominal trajectory

This controller uses [iLQR
optimization](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajopt.ilqr.html)
at every timestep during the execution resulting in a [model predictive
controller](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.mpc.ilqr.html). At
every timestep the first control command is returned to the simulation/hardware.

This controller uses iLQR optimization without a reference trajectory, the full optimization problem
is solved online. The goal state is kept fix.
