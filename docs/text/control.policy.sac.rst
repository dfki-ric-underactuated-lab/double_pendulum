Soft Actor Critic (SAC)
==============================

The SAC controller is a reinforcement learning-based control strategy 
that is particularly well-suited for scenarios with continuous action 
and state spaces. In such environments, where the agent has an infinite
range of actions to choose from and the system's states can be 
represented as continuous real values, the SAC controller proves to 
be effective.

In the context of the double pendulum swing-up and stabilization 
tasks, the actuators controlling the double pendulum can be 
adjusted to any value within the specified torque limit range. 
Additionally, the measurements of position and velocity obtained 
from the motors can also be represented continuously. 
The problem setup makes it suitable for SAC implementation.

SAC[1][2] optimizes a policy by maximizing the expected cumulative reward
obtained by the agent over time. This is achieved through an actor 
and critic structure [3].

The actor is responsible for selecting actions based on
the current policy in response to the observed state of the
environment. It is typically represented by a shallow neural
network that approximates the mapping between the input
state and the output probability distribution over actions.
SAC incorporates a stochastic policy in its actor part, which
encourages exploration and helps the agent improve policies.

The critic, on the other hand, evaluates the value of state-
action pairs. It estimates the expected cumulative reward that
the agent can obtain by following a certain policy. Typically,
the critic is also represented by a neural network that takes
state-action pairs as inputs and outputs estimated value.

In addition to the actor and critic, a central feature of SAC is entropy
regularization[4]. The policy is trained
to maximize a trade-off between expected return and entropy, which is a
measure of randomness in the action selection. If :math:`x` is a random
variable with a probability density function :math:`P`, the entropy
:math:`H` of :math:`x` is defined as:

.. math:: H(P) = \displaystyle \mathop{\mathbb{E}}_{x \sim P}[-\log P(x)]

By maximizing entropy, SAC encourages exploration and accelerates
learning. It also prevents the policy from prematurely converging to a
suboptimal solution. The trade-off between maximizing reward and
maximizing entropy is controlled through a parameter, :math:`\alpha`.
This parameter serves to balance the importance of exploration and
exploitation within the optimization problem. The optimal policy
:math:`\pi^*` can be defined as follows:

.. math::

   \pi^* = {arg}{\max_{\pi}}{\displaystyle
    \mathop{\mathbb{E}}_{\tau\sim\pi}}{\Bigg[{\sum_{t=0}^{\infty}}{\gamma^{t}}{\Big(R(s_t,a_t,s_{t+1})}+{\alpha}H(\pi(\cdot\mid{s_t}))\Big)\Bigg]}

During training, SAC learns a policy :math:`\pi_{\theta}` and two
Q-functions :math:`Q_{\phi_1} , Q_{\phi_2}` concurrently. The loss
functions for the two Q-networks are :math:`(i \in {1, 2})`:

.. math::

   L(\phi_i,D) = \displaystyle
     \mathop{\mathbb{E}}_{(s,a,r,s',d)\sim{D}}\bigg[\bigg(Q_{\phi_i}(s,a)-y(r,s',d)\bigg)^2\bigg],

where the temporal difference target :math:`y` is given by:

.. math::

   \begin{aligned}
     y(r,s',d) &= r + \gamma(1-d) \times \nonumber \\
     & \bigg(\displaystyle
     \mathop{\min}_{j=1,2}Q_{\phi_{targ,j}}(s',\tilde{a}')-\alpha\log
     {\pi_\theta}(\tilde{a}'\mid{s}')\bigg), \\
     \tilde{a}'&\sim{\pi_\theta}(\cdot\mid{s'})
   \end{aligned}

In each state, the policy :math:`\pi_\theta` should act to maximize the
expected future return :math:`Q` while also considering the expected
future entropy :math:`H`. In other words, it should maximize
:math:`V^\pi(s)`:

.. math::

   \begin{aligned}
    V^\pi(s) &= {\displaystyle \mathop{\mathbb{E}}_{a\sim\pi}[Q^\pi(s,a)]} +
    \alpha{H(\pi(\cdot\mid{s}))} \\
    &= {\displaystyle \mathop{\mathbb{E}}_{a\sim\pi}[Q^\pi(s,a)]} -
    \alpha{\log {\pi(a\mid{s})}}
   \end{aligned}

By employing an effective gradient-based optimization technique, the
parameters of both the actor and critic neural networks undergo updates,
subsequently leading to the adaptation of the policies themselves.

In conclusion, SAC’s combination of stochastic policies, exploration
through entropy regularization, value estimation, and gradient-based
optimization make it a well-suited algorithm for addressing the
challenges posed by continuous state and action spaces.

References
----------
- [1] Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum
  entropy deep reinforcement learning with a stochastic actor.
  " International conference on machine learning. PMLR, 2018.
- [2] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and 
  applications." arXiv preprint arXiv:1812.05905 (2018).
- [3] Konda, Vijay, and John Tsitsiklis. "Actor-critic algorithms.
  " Advances in neural information processing systems 12 (1999).
- [4] J. Achiam, “Spinning Up in Deep Reinforcement Learning,” 2018.
  url: `<https://spinningup.openai.com/en/latest/algorithms/sac.html>`__