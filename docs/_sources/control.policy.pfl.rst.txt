Partial Feedback Linearization (PFL)
====================================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

Partial Feedback Linearization (PFL) [1-3] is a
classical method from control theory. With PFL it is possible to provoke a
linear response in both joints of the double pendulum even if operated as a
pendubot or acrobot.
For an intuition of its functionality consider the manipulator equation for the
acrobot (:math:`u_1 \equiv 0`)

.. math::

  \left[ {\begin{array}{cc}
    M_{11} & M_{12} \\
    M_{21} & M_{22} \\
  \end{array} } \right]
  \left[ {\begin{array}{c}
    \ddot{q}_1 \\
    \ddot{q}_2 \\
  \end{array} } \right]
  +
  \left[ {\begin{array}{cc}
    C_{11} & C_{12} \\
    C_{21} & C_{22} \\
  \end{array} } \right]
  \left[ {\begin{array}{c}
    \dot{q}_1 \\
    \dot{q}_2 \\
  \end{array} } \right]
  +
  \left[ {\begin{array}{c}
    G_1 \\
    G_2 \\
  \end{array} } \right]
  +
  \left[ {\begin{array}{c}
    F_1 \\
    F_2 \\
  \end{array} } \right]
  -
  \left[ {\begin{array}{cc}
    0 & 0 \\
    0 & 1 \\
  \end{array} } \right]
  \left[ {\begin{array}{c}
    0 \\
    u_2 \\
  \end{array} } \right]
  = 0

The unactuated upper part of the vector equation can
be solved for the acceleration :math:`\ddot{q}_1` and then plugged into the lower
part of the equation. The control input :math:`u_2` can now be designed as PD control
with an energy term

.. math::

  u_2(\vect{x}) = -k_p(q_2 - q_2^{d}) - k_d\dot{q_2} + k_e(E - E^{d})\dot{q}_1
  \label{eq:pfl_acro_col}

with the desired configuration :math:`q_2^{d}` of the second link, the total
energy :math:`E`, the desired total energy :math:`E^{d}` and the gain parameters :math:`k_p,
k_d` and :math:`k_e`. The above described method is called collocated PFL.
Similarly, it is also possible to eliminate :math:`\ddot{q}_2` instead of
:math:`\ddot{q}_1` from the equations which is than called non-collocated PFL. Partial
feedback linearization for the pendubot can be done on the same way. The
collocated control law in this case reads

.. math::

  u_1(\vect{x}) = -k_p(q_1 - q_1^{d}) - k_d\dot{q}_1 + k_e(E - E^{d})\dot{q}_2.
  \label{eq:_pfl_pendu_col}

References
----------

- [1] M. W. Spong, “Swing up control of the acrobot using partial feedback
  linearization” IFAC Proceedings Volumes, vol. 27, no. 14, pp. 833–838,
  Sep. 1994, doi: 10.1016/S1474-6670(17)47404-0.
  url: `<https://www.sciencedirect.com/science/article/pii/S1474667017474040?via%3Dihub>`__
- [2] M. W. Spong, “The swing up control problem for the Acrobot,” IEEE Control
  Systems Magazine, vol. 15, no. 1, pp. 49–55, Feb. 1995, doi:
  10.1109/37.341864.
  url: `<https://ieeexplore.ieee.org/document/341864>`__
- [3] M. W. Spong, “Energy Based Control of a Class of Underactuated Mechanical
  Systems,” IFAC Proceedings Volumes, vol. 29, no. 1, pp. 2828–2832, Jun. 1996,
  doi: 10.1016/S1474-6670(17)58105-7.
  url: `<https://www.sciencedirect.com/science/article/pii/S1474667017581057?via%3Dihub>`__

