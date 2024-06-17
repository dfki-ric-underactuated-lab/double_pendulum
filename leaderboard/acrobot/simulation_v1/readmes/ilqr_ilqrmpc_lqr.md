# Iterative Linear Quadratic Regulator (iLQR) MPC Stabilization

## Trajectory Optimization

This controller uses a trajectory calculated with [iLQR
optimization](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajopt.ilqr.html).

## Trajectory Stabilization

For stabilizing the trajectory during the execution, this controller uses [iLQR
optimization](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajopt.ilqr.html)
at every timestep during the execution resulting in a [model predictive
controller](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.mpc.ilqr.html). At
every timestep the first control command is returned to the simulation/hardware.
The cost function penalizes deviations from the nominal trajectory.

## Final Stabilization

For stabilizing the final state, the unstable fix point, this controller uses
the [LQR
controller](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.policy.lqr.html).
