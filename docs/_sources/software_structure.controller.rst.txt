Controller Class
================

The controller class is central the this library as it has connections to
many other components. To ensure compatibility with all components, controllers
should always inherit from the abstract_controller class.
The abstract controller class can optionally do logging, state filtering,
friction compensation and gravity compensation.

Class Methods
-------------

Every specific controller, inheriting from the abstract controller class has
to implement a

.. code::

   controller.get_control_output_(self, x, t)

method, which should return an array-like object with shape=(2,) representing
the torques to be sent to the motors. The ``get_control_output_`` or more
precisely the ``get_control_output`` (without underscore) of the abstract
controller will be called by the simulator and the hardware control loop during
hardware experiments.
In addition to this function controllers can optionally have the methods

.. code::

   controller.set_parameters(self, ...)
   controller.set_goal(self, goal)
   controller.init_(self)
   controller.reset_(self)
   controller.save_(self, save_dir)
   controller.get_forecast(self)
   controller.get_init_trajectory(self)


``set_parameter`` can be used to set controller specific parameters and
``set_goal`` can be used to set the goal and maybe compute internal controller
properties which depend on the goal. 
The ``init_`` method will be called before the execution and ``reset_`` can
be used to reset parameters inside the controller. ``save_`` is used to store
controller parameters to be able retrace the controller properties after
execution. ``get_forecast`` can be implemented in predictive controllers to
return the predicted trajectory and ``get_init_trajectory`` can return the
initialy planned trajectory of the controller (e.g. the trajectory a controller
is supposed to stabilize). If the latter two trajectory returning methods are
implemented these trajectories can ba plotted in the animation funciton of the
simulator.

Logging, Filtering and Compensation
-----------------------------------

The abstract controller class has methods implemented for logging, state
filtering and friction/gravity compensation.

To use the state filtering one has to call ``set_filter_args`` method with the
desired filter parameters. The filter is then initialized during the
``controller.init`` and incoming state measurements are filtered before being
parsed to the ``get_control_output_`` method. By default no filtering is used.

Friction compensation can be turned on by calling ``set_friction_compensation``
with the desired friction parameters. Gravity compensation is activated with
``set_gravity_compensation`` with a plant object as parameter. Note that gravity
can only be fully compensated on the fully actuated double pendulum. Friction
and gravity compensation torques are both added to the torque returned by the
``get_control_output_`` method.

The controller internally logs the total controller torque in ``u_hist``, the
friciton compensation torque in ``u_fric_hist``, the gravity compensation torque
in ``u_grav_hist``, the state history in ``x_hist`` and the filtered states in
``xfilt_hist`` and they can be accesses via the controller object.

Writing your own controller
---------------------------

If you want to write your own controller you may want to use the template below.
The only method that is strictly necessary and has to be implemented is the
``get_control_output`` method. The other methods are optional.

.. code::

  from double_pendulum.controller.abstract_controller import AbstractController


  class ControllerTemplate(AbstractController):
      """Controller Template"""

      def __init__(self):
          super().__init__()

      def set_parameters(self):
          """
          Set controller parameters. Optional.
          Can be overwritten by actual controller.
          """
          pass

      def set_goal(self, x):
          """
          Set the desired state for the controller. Optional.
          Can be overwritten by actual controller.

          Parameters
          ----------
          x : array_like, shape=(4,), dtype=float,
              state of the double pendulum,
              order=[angle1, angle2, velocity1, velocity2],
              units=[rad, rad, rad/s, rad/s]
          """
          self.goal = x

      def init_(self):
          """
          Initialize the controller. Optional.
          Can be overwritten by actual controller.
          Initialize function which will always be called before using the
          controller.
          """
          pass

      def reset_(self):
          """
          Reset the Controller. Optional
          Can be overwritten by actual controller.
          Function to reset parameters inside the controller.
          """
          pass

      def get_control_output_(self, x, t=None):
          """
          The function to compute the control input for the double pendulum's
          actuator(s).
          Supposed to be overwritten by actual controllers. The API of this
          method should not be changed. Unused inputs/outputs can be set to None.

          Parameters
          ----------
          x : array_like, shape=(4,), dtype=float,
              state of the double pendulum,
              order=[angle1, angle2, velocity1, velocity2],
              units=[rad, rad, rad/s, rad/s]
          t : float, optional
              time, unit=[s]
              (Default value=None)

          Returns
          -------
          array_like
              shape=(2,), dtype=float
              actuation input/motor torque,
              order=[u1, u2],
              units=[Nm]
          """
          u = [0.0, 0.0]
          return u

      def save_(self, save_dir):
          """
          Save controller parameters. Optional
          Can be overwritten by actual controller.

          Parameters
          ----------
          save_dir : string or path object
              directory where the parameters will be saved
          """
          pass

      def get_forecast(self):
          """
          Get a forecast trajectory as planned by the controller. Optional.
          Can be overwritten by actual controller.

          Returns
          -------
          list
              Time array
          list
              X array
          list
              U array
          """
          return [], [], []

      def get_init_trajectory(self):
          """
          Get an initial (reference) trajectory used by the controller. Optional.
          Can be overwritten by actual controller.

          Returns
          -------
          list
              Time array
          list
              X array
          list
              U array
          """
          return [], [], []
