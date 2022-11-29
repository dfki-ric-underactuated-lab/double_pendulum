Linear Quadratic Regulator (LQR)
================================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

The linear quadratic regulator (LQR) controller is a well
established and widespread optimal controller which acts on a linear system
:math:`\dvect{x} = \mat{A} \vect{x} + \mat{B} \vect{u}` and an objective which
is specified by a quadratic, instantaneous cost function :math:`J = \vect{x}^T
\mat{Q} \vect{x} + \vect{u}^T \mat{R} \vect{u}` with the symmetric and positive
definite matrices :math:`\mat{Q} = \mat{Q}^T \succeq 0` and :math:`\mat{R} =
\mat{R}^T \succ 0`.
This allows for reducing the Hamilton-Jacobi-Bellman (HJB) equation, whose
solution is the optimal cost-to-go, from which the optimal policy can be
inferred, to the algebraic Riccati equation

.. math::

    \mat{S}\mat{A} + \mat{A}^T\mat{S} -
    \mat{S}\mat{B}\mat{R}^{-1}\mat{B}^T\mat{S} + \mat{Q} = 0

for which good numerical solvers exist to find the optimal cost-to-go matrix :math:`\mat{S}`.
The optimal policy obtained is

.. math::

     \vect{u}(\vect{x}) = -\mat{R}^{-1}\mat{B}^{T}\mat{S}\vect{x} = -\mat{K}\vect{x}.

In order to use an LQR controller for stabilizing the double pendulum on the
top, the dynamics have to be linearised around the top position
:math:`\vect{x}^{d} = [\pi, 0, 0, 0]` and :math:`\vect{u}^{d} = [0, 0]`:

.. math::

    \mat A = \left. \frac{\partial \vect{f}(\vect{x}, \vect{u})}{\partial
    \vect{x}}\right|_{\vect{x}=\vect{x}^{d}, \vect{u}=\vect{u}^{d}}, \mat B =
    \left. \frac{\partial \vect{f}(\vect{x}, \vect{u})}{\partial
    \vect{u}}\right|_{\vect{x}=\vect{x}^{d}, \vect{u}=\vect{u}^{d}}

and the state and actuation have to be expressed in relative coordinates
:math:`\tilde{\vect{x}} = \vect{x} - \vect{x}^{d}`, :math:`\tilde{\vect{u}} =
\vect{u} - \vect{u}^{d}`.

Region of Attraction (RoA)
--------------------------

For dynamical systems, the Region of Attraction
(RoA) :math:`\mathcal{B}` around a fixed point :math:`\vect{x}^{\star}` describes the set
of initial states for which :math:`\vect{x} \rightarrow \vect{x}^{\star}` as 
:math:`t\rightarrow \infty`.
Direct computation of this set is often not possible. However, it can be
estimated by considering the sublevel set of a Lyapunov function :math:`V(\vect{x})`
[1].
When using LQR to stabilize the system around :math:`\vect{x}^{\star}`, the
cost-to-go can serve as a quadratic Lyapunov function [2].
In this case, the estimated RoA can be written as:

.. math::

  \mathcal{B}_{\text{est}} = \left \{ \vect{x} \vert \vect{x}^{T} \mat{S} \vect{x} \leq \rho \right \}

Where :math:`\rho` is a scalar that can be estimated using either
probabilistic [3] or optimization based methods [4].


For further reading we refer to these lecture notes [2].

References
----------

- [1] H. K. Khalil, Nonlinear Systems, 3rd ed. Upper Saddle River, N.J:
  Prentice Hall, 2002
- [2] R. Tedrake, Underactuated Robotics, 2022. (Online)
  url: `<http://underactuated.mit.edu>`__
- [3] E. Najafi, R. Babuška, and G. A. D. Lopes, “A fast sampling method for
  estimating the domain of attraction,” Nonlinear Dynamics, vol. 86, no. 2, pp.
  823–834, Oct. 2016.
  url: `<https://link.springer.com/article/10.1007/s11071-016-2926-7>`__
- [4]  P. Parrilo, “Structured semidefinite programs and semialgebraic ge-
  ometry methods in robustness and optimization,” Ph.D. dissertation,
  California Institute of Technology, Pasadena, California, 2000.
  url: `<https://www.proquest.com/openview/ff5fe1a4311720ae2dad28ddc1d22cf8/1?cbl=18750&diss=y&pq-origsite=gscholar&parentSessionId=MjXEze6vRVD%2BeSjkr1UEy6Zldtg74txylCbk173fanA%3D>`__
