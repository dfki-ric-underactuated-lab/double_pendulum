# Time-varying LQR (TVLQR) with iLQR trajectory

## Trajectory Optimization

This controller uses a trajectory calculated with [iLQR
optimization](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajopt.ilqr.html).

## Trajectory Stabilization

For stabilizing the trajectory during the execution, this controller uses
[TVLQR](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.trajstab.tvlqr.html).

The TVLQR stabilization suffices to stabilize the unstable fixpoint in the end.

## Final Stabilization

For stabilizing the final state, the unstable fix point, this controller uses
the [LQR
controller](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.policy.lqr.html).
