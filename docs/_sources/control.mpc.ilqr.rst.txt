Iterative Linear Quadratic Regulator (iLQR) MPC
===============================================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

iLQR is a shooting method and as such has the property that all trajectories
during the optimization process are physically feasible. So even when stopped
before convergence, the solution is not inconsistent. This has the advantage
that iLQR can be used in a Model Predictive Control (MPC) ansatz.
For this the optimization is performed online and at every
time step the first control input :math:`\vect{u}_0` is executed. For the next time
step the previous solution is used to warm start the next optimization step.

When used for stabilizing a nominal trajectory, the iLQR optimization
problem is solved with time varying desired states
:math:`\vect{x}^{d} = \vect{x}^{d}(t)` and inputs :math:`\vect{u}^{d} =
\vect{u}^{d}(t)`.

When used as free optimization, the full optimization problem is solved online.
The goal state :math:`\vect{x}^d` is kept fix.

