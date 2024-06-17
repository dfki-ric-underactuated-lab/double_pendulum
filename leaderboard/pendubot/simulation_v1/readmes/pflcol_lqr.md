# Collocated Partial Feedback Linearization (PFL) with Energy Shaping

# Swingup

This controller uses [Partial Feedback
Linearization](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.policy.pfl.html)
to provoke a linear response in both joints of the double pendulum even if
operated as a pendubot or acrobot. Energy shaping control and PD control is
added on top of PFL to reach the goal configuration.

## Final Stabilization

For stabilizing the final state, the unstable fix point, this controller uses
the [LQR
controller](https://dfki-ric-underactuated-lab.github.io/double_pendulum/control.policy.lqr.html).
