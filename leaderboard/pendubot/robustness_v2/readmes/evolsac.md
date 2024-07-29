# Evolutionary SAC

## Trajectory Learning, Optimization and Stabilization

This controller uses a trajectory dictated by the policy learned in the following way:

  1. SAC training with loose surrogate reward
  2. SAC training with stricter surrogate reward
  3. SNES training with challenge reward + injected noise in the action

The controller uses the final policy network
