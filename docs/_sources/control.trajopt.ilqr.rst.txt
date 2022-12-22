Iterative Linear Quadratic Regulator (iLQR)
===========================================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

Iterative LQR (iLQR) [1] is an extension of LQR to non-
linear dynamics. The LQR uses the fixed point linearised
dynamics for the entire state space and hence is only useful
as long as the linearization error is small. In contrast to LQR,
iLQR linearises the dynamics for every given state at each
time step and can deal with nonlinear dynamics at the cost
of being only able to optimize over a finite time horizon.

As a trajectory optimization method iLQR solves the
following optimization problem for a sequence of :math:`N` control inputs:

.. math::

  \min_{\vect{u}_0, \vect{u}_1, \ldots, \vect{u}_{N-1}} &\vect{x}_N^T
  \mat{Q}_{f} \vect{x}_N + \sum_{i=0}^{N-1} \vect{x}_i^T \mat{Q} \vect{x}_i +
  \vect{u}_i^T \mat{R} \vect{u}_i \label{eq:ilqr_opt}\\ \text{subject to}: &
  \hspace{1cm} \vect{x}_{i+1} = f_{discrete}(\vect{x}_{i}, \vect{u}_i)

where a start state :math:`\vect{x}_0` is set beforehand. :math:`\mat{Q}_f`,
:math:`\mat{Q}` and :math:`\mat{R}` are cost matrices penalizing the final
state, intermediate states and the control input respectively.
:math:`f_{discrete}` is the discretization of the system dynamics.
:math:`\vect{x}` and :math:`\vect{u}` can also be expressed in relative
coordinates :math:`\tilde{\vect{x}}`, :math:`\tilde{\vect{u}}`.



References
----------

- [1]  L. Weiwei and E. Todorov, “Iterative Linear Quadratic Regulator Design
  for Nonlinear Biological Movement Systems.” International Conference on
  Informatics in Control, Automation and Robotics, pp.  222–229, 2004.
  url: `<https://www.scitepress.org/Link.aspx?doi=10.5220/0001143902220229>`__
