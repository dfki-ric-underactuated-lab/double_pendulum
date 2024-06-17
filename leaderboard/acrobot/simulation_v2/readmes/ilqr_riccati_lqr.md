# iLQR Riccati Gains

## Trajectory Optimization

This controller uses a trajectory calculated with [iLQR
optimization](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajopt.ilqr.html).

## Trajectory Stabilization

For stabilizing the trajectory during the execution, this controller uses the
[Riccati feedback
gains](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajstab.riccati.html)
that were used during the offline iLQR trajectory optimization to react to
deviations from the nominal trajectory.

## Final Stabilization

For stabilizing the final state, the unstable fix point, this controller uses
the [LQR
controller](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.policy.lqr.html).
