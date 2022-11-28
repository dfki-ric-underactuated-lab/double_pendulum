System Identification
=====================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

For the identification of the 15 model parameters, we fix the natural, provided
and easily measurable parameters :math:`g, g_r, l_1` and :math:`l_2` and
consider the independent model parameters

.. math::

  m_1 r_1,\, m_2 r_2,\, m_2,\, I_1,\, I_2,\, I_r,\, b_1,\, b_2,\, c_{f1},\,
  c_{f2}.

The goal is to identify the parmaters of the dynamic matrices in the
manipulator equation

.. math::

  \mat{M} \ddot{\vect{q}} + \mat{C}(\vect{q}, \dot{\vect{q}}) \dot{\vect{q}} -
  \mat{G}(\vect{q}) + \mat{F}(\dot{\vect{q}}) - \mat{D} \vect{u} = 0

By executing excitation trajectories on the real hardware, data tuples of the
form :math:`(\vect{q}, \dot{\vect{q}}, \ddot{\vect{q}}, \vect{u})` can be
recorded.  For finding the best system parameters, one can make use of the fact
that the dynamics matrices :math:`\mat{M}, \mat{C}, \mat{G}` and
:math:`\mat{F}` are linear in the independent model parameters and perform a
least squares optimization for the dynamics equation on the recorded data.

