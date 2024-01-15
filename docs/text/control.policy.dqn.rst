Deep-Q Learning (DQN)
==============================

The DQN controller is a reinforcement learning-based control strategy 
that is particularly well-suited for scenarios with discrete action spaces. 
However, discretizing the action space can be an option when its dimensionality is small.
The action space for pendubot and acrobot is only composed of one dimension.   

In the context of the double pendulum swing-up and stabilization 
tasks, the actuators controlling the double pendulum can be 
adjusted to any value within the specified torque limit range. 
Additionally, the measurements of position and velocity obtained 
from the motors can also be represented continuously. 
The problem setup makes it suitable for DQN implementation.

The idea of DQN is to learn an action-value function from which a greedy
policy would yield the highest possible sum of discounted rewards.
To learn such a function, this method uses the optimal Bellman operator. 
This operator is a contracting mapping, meaning that the successive iterations of
this operator lead to its fixed point. The theory guarantees
that this fixed point is the optimal action-value function cor-
responding to the optimal policy i.e., the policy yielding max-
imum reward.

References
----------
- [1] V. Mnih, K. Kavukcuoglu, D. Silver, et al., “Human-
  level control through deep reinforcement learning,” na-
  ture, vol. 518, no. 7540, pp. 529–533, 2015.
