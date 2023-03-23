Controller Robustness
=====================

When transferring controllers from simulation to real hardware many effects that
are not present in simulation may influence the behavior. Often it is the case
that controllers are tuned in simulation and are capable of high quality
performances while in real system experiments they fail to achieve the desired
results. This phenomenon is commonly referred to as simulation-reality gap.
In order to study the transferability of controllers, we conduct robustness 
tests in simulation.
The robustness tests quantify how well the controllers
perform under the following conditions:

- **Model inaccuracies**: The model parameters, that have been determined with
  system identification, will never be perfectly accurate. To asses
  inaccuracies in these parameters, we vary the independent model parameters
  one at the time in the simulator while using the original model parameters in
  the controller.
- **Measurement noise**: The controllers' outputs depend on the measured system
  state.  In the case of the QDDs, the online velocity measurements are noisy.
  Hence, it is important for the transferability that a controller can handle
  at least this amount of noise in the measured data.  The controllers are
  tested with and without a low-pass noise filter.
- **Torque noise**: Not only the measurements are noisy, but also the torque that
  the controller outputs is not always exactly the desired value. 
- **Torque response**: The requested torque of the controller will in general not
  be constant but change during the execution. The motor, however, is sometimes
  not able to react immediately to large torque changes and will instead
  overshoot or undershoot the desired value.  This behavior is modelled by
  applying the torque :math:`\tau = \tau_{t-1} + k_{resp} (\tau_{des} -
  \tau_{t-1})` instead of the desired torque :math:`\tau_{des}`.  Here,
  :math:`\tau_{t-1}` is the applied motor torque from the last time step and
  :math:`k_{resp}` is the factor which scales the responsiveness.
  :math:`k_{resp} = 1` means the torque response is perfect while
  :math:`k_{resp} \neq 1` means the motor is over/undershooting the desired
  torque. 
- **Time delay**: When operating on a real system there will always be time delays
  due to communication and reaction times. 
