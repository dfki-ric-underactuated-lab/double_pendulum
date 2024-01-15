Control Methods
===============

There are many methods to control the double pendulum/acrobot/pendubot system.
Here, we categorised them in

- Trajectory Optimization
- Trajectory Stabilization
- Policy-based Control
- Model Predictive Control

even though the methods can build upon each other. E.g. a trajectory can be
computed with a trajectory optimization algorithm and then be stabilized with
a trajectory stabilization algorithm.

.. toctree::
   :maxdepth: 2

   control.trajopt.rst
   control.trajstab.rst
   control.policy.rst
   control.mpc.rst

