# Variational Model Predictive Path Integral Control

## Sampling and Weighed Sum

This controller uses the [Path Integral Control](https://ieeexplore.ieee.org/document/7487277/) to sample trajectories and weigh them to find the optimal control. The rollouts are computed using the [variational integrator](https://courses.cms.caltech.edu/cs171/assignments/hw6/hw6-notes/notes-hw6.html), which allows to have much longer timespan why keeping the computations in the real time. 

