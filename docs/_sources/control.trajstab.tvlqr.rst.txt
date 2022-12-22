Time Varying Linear Quadrativ Regulator (TVLQR)
===============================================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

Time-Varying LQR (TVLQR) is another extension to the regular LQR algorithm and
can be used to stabilize a nominal trajectory (:math:`\vect{x}^{d}(t),
\vect{u}^{d}(t)`). For this, the LQR formalization is used for time-varying
linear dynamics

.. math::

  \dvect{x} = \mat{A}(t) (\vect{x} - \vect{x}^{d}(t)) + \mat{B}(t) (\vect{u} - \vect{u}^{d}(t))

which requires to linearise~(\ref{eq:dyn}) at all steps around (:math:`\vect{x}^{d}(t),
\vect{u}^{d}(t)`). This results in the optimal policy at time :math:`t`

.. math::

  \vect{u}(\vect{x}, t) = \vect{u}^{d} - \mat{K}(t) (\vect{x} - \vect{x}^{d}(t)).


For further reading we refer to these lecture notes [1].

References
----------

- [1] R. Tedrake, Underactuated Robotics, 2022. (Online)
  url: `<http://underactuated.mit.edu>`__
