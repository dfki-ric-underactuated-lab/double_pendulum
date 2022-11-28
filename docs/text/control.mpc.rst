Model Predictive Control (MPC)
==============================

The idea of model predictive control is to solve the trajectory optimization
problem at every timestep again. At each step the optimization is warm started
with the solution of the previous timestep and update is computed and the first
control command from the trajectory is executed.

MPC can either be used to stabilize a reference trajectory (the costs are
computed in refernce to that trajectory) or as a free method where the full
optimization problem is solved online.

.. toctree::
   :maxdepth: 1

   control.mpc.ilqr.rst
