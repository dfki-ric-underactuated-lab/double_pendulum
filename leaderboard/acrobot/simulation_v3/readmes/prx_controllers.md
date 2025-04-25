# PRX controllers

## Analytical Trajectory follower 
This controller uses a precomputed trajectory to compute gains (in an iLQR fashion) to follow the trajectory

## Stabilization
Classic LQR control (u(x) = -K*(x - goal)) around the goal (pi,0,0,0)
