Policy-based Control
====================

.. math::

      \newcommand{\vect}[1]{\boldsymbol{#1}}
      \newcommand{\dvect}[1]{\dot{\boldsymbol{#1}}}
      \newcommand{\ddvect}[1]{\ddot{\boldsymbol{#1}}}
      \newcommand{\mat}[1]{\boldsymbol{#1}}

Policy based controllers do not aim to follow a singular trajectory. Instead,
they define a funtion(the policy) :math:`\pi` over the entire state space and
return a control signal according to this policy:

.. math::

   \vect{u} = \pi(\vect{x})

the policy can be either analytically defined by reasoning about the system
(e.g. about the system's energy) or it can be learned with reinforcement
learning.

.. toctree::
   :maxdepth: 1

   control.policy.lqr.rst
   control.policy.pfl.rst
   control.policy.sac.rst
   control.policy.dqn.rst
   .. control.energyXin.rst
