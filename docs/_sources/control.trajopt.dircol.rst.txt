Direct Collocation
==================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

Direct collocation [1] is a trajectory optimization method which belongs to the
collocation methods. The optimal control problem is transformed into a
mathematical programming problem by transcribing the trajectory with :math:`N`
knot points :math:`\{\vect{x}_i\}`.  For the optimization a loss function
:math:`L(\{\vect{x}_i\}, \{\vect{u}_i\})` is minimized with the knot points and
the control inputs :math:`\{\vect{u}_i\}` as decision variables. The
formulation of the optimization problem to compute a trajectory from an initial
state :math:`x_{init}` to a goal state :math:`x_{goal}` while obeying control
limits :math:`(u_{1, max}, u_{2, max})` is

.. math::

  \min_{\vect{x}_0, \ldots, \vect{x}_{N}, \vect{u}_0, \ldots, \vect{u}_{N-1}}
  &L(\{\vect{x}_i\}, \{\vect{u}_i\})\\
  \text{subject to}:  \hspace{1cm}& \vect{x}_{i+1} = f_{discrete}(\vect{x}_{i},
  \vect{u}_i) \quad \forall i\\
  \hspace{1cm}& x_0 = x_{init} \\
  \hspace{1cm}& x_N = x_{goal} \\
  \hspace{1cm}& |u_{1,i}| \leq u_{1,max} \quad \forall i \\
  \hspace{1cm}& |u_{2,i}| \leq u_{2,max} \quad \forall i

The numerical solution can be obtained with Sequential Quadratic Programming (SQP).
In practice, it can be useful to interpolate the trajectory between knot points
as well as between controls with polynomials and then using the parameters of
these piecewise polynomials as decision variables instead. This allows to use
fewer knot points and still obtain consistent trajectories.

References
----------

- [1] Hargraves, Charles R., and Stephen W. Paris. "Direct trajectory
  optimization using nonlinear programming and collocation." Journal of
  guidance, control, and dynamics 10.4 (1987): 338-342.
  url: `<https://arc.aiaa.org/doi/abs/10.2514/3.20223>`__
